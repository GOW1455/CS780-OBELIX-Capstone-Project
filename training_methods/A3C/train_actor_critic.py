from __future__ import annotations

import argparse
import importlib.util
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.distributions import Categorical

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


class ActorCritic(nn.Module):
    def __init__(self, in_dim: int = 18, n_actions: int = 5, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value


class SharedAdam(optim.Adam):
    """Adam optimizer with shared states for multiprocessing A3C."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


def import_obelix(obelix_py: str):
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def copy_grads_to_global(local_model: ActorCritic, global_model: ActorCritic) -> None:
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        if local_param.grad is None:
            continue
        global_param._grad = local_param.grad


def eps_by_step(step: int, eps_start: float, eps_end: float, eps_decay_steps: int) -> float:
    if step >= eps_decay_steps:
        return eps_end
    frac = step / float(eps_decay_steps)
    return eps_start + frac * (eps_end - eps_start)


def worker_process(
    worker_id: int,
    args,
    global_model: ActorCritic,
    optimizer: SharedAdam,
    episode_counter,
    counter_lock,
    global_step_counter,
    step_lock,
):
    torch.set_num_threads(1)
    seed = args.seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    OBELIX = import_obelix(args.obelix_py)
    local_model = ActorCritic()

    while True:
        with counter_lock:
            if episode_counter.value >= args.episodes:
                break
            episode_idx = int(episode_counter.value)
            episode_counter.value += 1

        local_model.load_state_dict(global_model.state_dict())

        env_seed = args.seed + episode_idx * args.num_workers + worker_id
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=env_seed,
        )

        state = env.reset(seed=env_seed)
        done = False
        ep_return = 0.0
        ep_steps = 0
        last_eps = args.eps_start

        while not done and ep_steps < args.max_steps:
            log_probs = []
            values = []
            rewards = []
            entropies = []

            for _ in range(args.t_max):
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                logits, value = local_model(state_t)
                probs = torch.softmax(logits.squeeze(0), dim=-1)
                dist = Categorical(probs)

                with step_lock:
                    current_step = int(global_step_counter.value)
                    global_step_counter.value += 1
                last_eps = eps_by_step(current_step, args.eps_start, args.eps_end, args.eps_decay_steps)

                if np.random.random() < last_eps:
                    action_idx = int(np.random.randint(len(ACTIONS)))
                    action_t = torch.tensor(action_idx, dtype=torch.int64)
                else:
                    action_t = dist.sample()
                    action_idx = int(action_t.item())

                log_probs.append(dist.log_prob(action_t))
                values.append(value.squeeze(0))
                entropies.append(dist.entropy())
                next_state, reward, done = env.step(ACTIONS[action_idx], render=True)

                rewards.append(float(reward))
                ep_return += float(reward)
                ep_steps += 1
                state = next_state

                if done or ep_steps >= args.max_steps:
                    break

            if done:
                bootstrap_value = torch.tensor(0.0)
            else:
                with torch.no_grad():
                    next_state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    _, next_value = local_model(next_state_t)
                    bootstrap_value = next_value.squeeze(0)

            returns = []
            running_return = bootstrap_value
            for r in reversed(rewards):
                running_return = torch.tensor(r, dtype=torch.float32) + args.gamma * running_return
                returns.insert(0, running_return)

            returns_t = torch.stack(returns)
            values_t = torch.stack(values)
            log_probs_t = torch.stack(log_probs)
            entropy_t = torch.stack(entropies)

            advantage = returns_t - values_t
            policy_loss = -(log_probs_t * advantage.detach()).mean()
            value_loss = 0.5 * advantage.pow(2).mean()
            entropy_loss = -args.entropy_coef * entropy_t.mean()
            loss = policy_loss + args.value_coef * value_loss + entropy_loss

            optimizer.zero_grad()
            local_model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(local_model.parameters(), args.grad_clip)
            copy_grads_to_global(local_model, global_model)
            optimizer.step()

            local_model.load_state_dict(global_model.state_dict())

        if (episode_idx + 1) % args.log_every == 0:
            print(
                f"[worker {worker_id}] episode={episode_idx + 1}/{args.episodes} "
                f"return={ep_return:.1f} steps={ep_steps} eps={last_eps:.3f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, default="./obelix_change.py")
    ap.add_argument("--out", type=str, default="ac_weights.pth")
    ap.add_argument("--episodes", type=int, default=1200)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--value_coef", type=float, default=0.5)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--eps_start", type=float, default=0.5)
    ap.add_argument("--eps_end", type=float, default=0.02)
    ap.add_argument("--eps_decay_steps", type=int, default=200000)
    ap.add_argument("--t_max", type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    global_model = ActorCritic()
    if os.path.exists(args.out):
        global_model.load_state_dict(torch.load(args.out, map_location="cpu"))
    global_model.share_memory()

    optimizer = SharedAdam(global_model.parameters(), lr=args.lr)
    episode_counter = mp.Value("i", 0)
    counter_lock = mp.Lock()
    global_step_counter = mp.Value("i", 0)
    step_lock = mp.Lock()

    print(f"Training A3C agent with {args.num_workers} workers...")
    start_time = time.time()

    workers = []
    for worker_id in range(args.num_workers):
        p = mp.Process(
            target=worker_process,
            args=(
                worker_id,
                args,
                global_model,
                optimizer,
                episode_counter,
                counter_lock,
                global_step_counter,
                step_lock,
            ),
        )
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    torch.save(global_model.state_dict(), args.out)
    elapsed = time.time() - start_time
    print(f"Saved: {args.out} | elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
