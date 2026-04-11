from __future__ import annotations

import argparse
import importlib.util
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
STATE_DIM = 18
ACTION_DIM = 5


class ActorCritic(nn.Module):
    def __init__(self, in_dim: int = STATE_DIM, n_actions: int = ACTION_DIM, hidden_dim: int = 256):
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


class RewardModel(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(action_idx.long(), num_classes=ACTION_DIM).float()
        x = torch.cat([state, one_hot], dim=-1)
        return self.net(x).squeeze(-1)


@dataclass
class Segment:
    states: np.ndarray
    actions: np.ndarray
    score: float


def import_obelix(obelix_py: str):
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


def choose_action(policy: ActorCritic, state: np.ndarray, rng: np.random.Generator) -> int:
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, _ = policy(state_t)
        probs = torch.softmax(logits.squeeze(0), dim=-1).cpu().numpy()
    probs_sum = float(np.sum(probs))
    if not np.isfinite(probs_sum) or probs_sum <= 0.0:
        probs = np.ones(ACTION_DIM, dtype=np.float32) / ACTION_DIM
    else:
        probs = probs / probs_sum
    return int(rng.choice(ACTION_DIM, p=probs))


def collect_segments(args, policy: ActorCritic, seed_offset: int = 0) -> list[Segment]:
    OBELIX = import_obelix(args.obelix_py)
    rng = np.random.default_rng(args.seed + seed_offset)
    segments: list[Segment] = []

    for ep in range(args.pref_episodes):
        env_seed = args.seed + seed_offset + ep
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
        seg_states: list[np.ndarray] = []
        seg_actions: list[int] = []
        seg_rewards: list[float] = []

        for _ in range(args.max_steps):
            a = choose_action(policy, state, rng)
            next_state, reward, done = env.step(ACTIONS[a], render=False)

            seg_states.append(np.asarray(state, dtype=np.float32))
            seg_actions.append(a)
            seg_rewards.append(float(reward))

            if len(seg_states) == args.segment_len:
                segments.append(
                    Segment(
                        states=np.stack(seg_states).astype(np.float32),
                        actions=np.asarray(seg_actions, dtype=np.int64),
                        score=float(np.sum(seg_rewards)),
                    )
                )
                seg_states.clear()
                seg_actions.clear()
                seg_rewards.clear()

            state = next_state
            if bool(done):
                break

    return segments


def build_preference_pairs(segments: list[Segment], pair_count: int, seed: int):
    if len(segments) < 2:
        raise RuntimeError("Not enough segments collected to build preferences.")

    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(pair_count):
        i, j = rng.choice(len(segments), size=2, replace=False)
        s1, s2 = segments[int(i)], segments[int(j)]
        label = 1.0 if s1.score >= s2.score else 0.0
        pairs.append((s1, s2, label))
    return pairs


def train_reward_model(args, reward_model: RewardModel, pairs) -> None:
    optimizer = optim.Adam(reward_model.parameters(), lr=args.reward_lr)
    bce = nn.BCEWithLogitsLoss()

    reward_model.train()
    for epoch in range(args.reward_epochs):
        random.shuffle(pairs)
        epoch_loss = 0.0

        for start in range(0, len(pairs), args.reward_batch_size):
            batch = pairs[start : start + args.reward_batch_size]
            if not batch:
                continue

            logits_list = []
            labels_list = []

            for s1, s2, label in batch:
                st1 = torch.tensor(s1.states, dtype=torch.float32)
                ac1 = torch.tensor(s1.actions, dtype=torch.int64)
                st2 = torch.tensor(s2.states, dtype=torch.float32)
                ac2 = torch.tensor(s2.actions, dtype=torch.int64)

                score1 = reward_model(st1, ac1).sum()
                score2 = reward_model(st2, ac2).sum()
                logits_list.append(score1 - score2)
                labels_list.append(label)

            logits = torch.stack(logits_list)
            labels = torch.tensor(labels_list, dtype=torch.float32)
            loss = bce(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        print(f"[reward model] epoch={epoch + 1}/{args.reward_epochs} loss={epoch_loss:.4f}")


def finetune_with_reward_model(args, policy: ActorCritic, reward_model: RewardModel) -> None:
    OBELIX = import_obelix(args.obelix_py)
    optimizer = optim.Adam(policy.parameters(), lr=args.rl_lr)
    rng = np.random.default_rng(args.seed + 10_000)

    reward_model.eval()
    policy.train()

    for ep in range(args.rl_episodes):
        env_seed = args.seed + 20_000 + ep
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
        ep_env_return = 0.0
        ep_model_return = 0.0

        for _ in range(args.max_steps):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            logits, value = policy(state_t)
            probs = torch.softmax(logits.squeeze(0), dim=-1)
            dist = Categorical(probs)

            if rng.random() < args.eps:
                action_idx = int(rng.integers(ACTION_DIM))
                action_t = torch.tensor(action_idx, dtype=torch.int64)
            else:
                action_t = dist.sample()
                action_idx = int(action_t.item())

            log_prob = dist.log_prob(action_t)
            entropy = dist.entropy()

            next_state, env_reward, done = env.step(ACTIONS[action_idx], render=False)
            ep_env_return += float(env_reward)

            with torch.no_grad():
                r_hat = reward_model(
                    torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                    torch.tensor([action_idx], dtype=torch.int64),
                ).squeeze(0)
                next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                _, next_value = policy(next_state_t)
                train_reward = args.reward_mix * torch.tensor(float(env_reward), dtype=torch.float32) + (1.0 - args.reward_mix) * r_hat
                target = train_reward
                if not bool(done):
                    target = target + args.gamma * next_value.squeeze(0)

            ep_model_return += float(train_reward.item())
            advantage = target - value.squeeze(0)

            actor_loss = -(log_prob * advantage.detach())
            critic_loss = 0.5 * advantage.pow(2)
            entropy_loss = -args.entropy_coef * entropy
            loss = actor_loss + args.value_coef * critic_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()

            state = next_state
            if bool(done):
                break

        if (ep + 1) % args.log_every == 0:
            print(
                f"[rlhf fine-tune] episode={ep + 1}/{args.rl_episodes} "
                f"env_return={ep_env_return:.1f} model_return={ep_model_return:.1f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, default="./obelix_change.py")
    ap.add_argument("--base_weights", type=str, default="ac_weights.pth")
    ap.add_argument("--out", type=str, default="ac_weights_rlhf.pth")
    ap.add_argument("--reward_model_out", type=str, default="reward_model.pth")

    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=1000)

    ap.add_argument("--pref_episodes", type=int, default=80)
    ap.add_argument("--segment_len", type=int, default=25)
    ap.add_argument("--pair_count", type=int, default=2000)
    ap.add_argument("--reward_epochs", type=int, default=8)
    ap.add_argument("--reward_batch_size", type=int, default=32)
    ap.add_argument("--reward_lr", type=float, default=1e-3)

    ap.add_argument("--rl_episodes", type=int, default=300)
    ap.add_argument("--rl_lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--value_coef", type=float, default=0.5)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--reward_mix", type=float, default=0.2)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policy = ActorCritic()
    if os.path.exists(args.base_weights):
        policy.load_state_dict(torch.load(args.base_weights, map_location="cpu"), strict=False)
    else:
        print(f"Warning: base weights not found at {args.base_weights}; starting from current random policy.")

    print("Collecting preference segments from current policy...")
    segments = collect_segments(args, policy)
    print(f"Collected segments: {len(segments)}")

    print("Building preference pairs...")
    pairs = build_preference_pairs(segments, args.pair_count, seed=args.seed + 1)
    print(f"Preference pairs: {len(pairs)}")

    reward_model = RewardModel()
    print("Training reward model...")
    train_reward_model(args, reward_model, pairs)

    torch.save(reward_model.state_dict(), args.reward_model_out)
    print(f"Saved reward model: {args.reward_model_out}")

    print("Fine-tuning policy with learned reward model...")
    finetune_with_reward_model(args, policy, reward_model)

    torch.save(policy.state_dict(), args.out)
    print(f"Saved RLHF policy: {args.out}")


if __name__ == "__main__":
    main()
