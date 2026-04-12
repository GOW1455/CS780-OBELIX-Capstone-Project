from __future__ import annotations

import argparse
import importlib.util
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
STATE_DIM = 18
ACTION_DIM = 5


class DQN(nn.Module):
    def __init__(self, in_dim: int = STATE_DIM, n_actions: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class Replay:
    def __init__(self, cap: int = 100_000):
        self.buf: Deque[Transition] = deque(maxlen=cap)

    def add(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch: int):
        idx = np.random.choice(len(self.buf), size=batch, replace=False)
        items = [self.buf[i] for i in idx]
        s = np.stack([it.s for it in items]).astype(np.float32)
        a = np.array([it.a for it in items], dtype=np.int64)
        r = np.array([it.r for it in items], dtype=np.float32)
        s2 = np.stack([it.s2 for it in items]).astype(np.float32)
        d = np.array([it.done for it in items], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)


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


def sample_action_from_q(q_values: np.ndarray, rng: np.random.Generator, temperature: float = 1.0) -> int:
    q_values = np.asarray(q_values, dtype=np.float32)
    temp = max(float(temperature), 1e-6)
    shifted = (q_values - np.max(q_values)) / temp
    probs = np.exp(shifted)
    probs_sum = float(np.sum(probs))
    if not np.isfinite(probs_sum) or probs_sum <= 0.0:
        probs = np.ones(ACTION_DIM, dtype=np.float32) / ACTION_DIM
    else:
        probs = probs / probs_sum
    return int(rng.choice(ACTION_DIM, p=probs))


def collect_segments(args, q: DQN) -> list[Segment]:
    OBELIX = import_obelix(args.obelix_py)
    rng = np.random.default_rng(args.seed)
    segments: list[Segment] = []

    q.eval()
    for ep in range(args.pref_episodes):
        env_seed = args.seed + ep
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=env_seed,
        )
        s = env.reset(seed=env_seed)

        seg_states: list[np.ndarray] = []
        seg_actions: list[int] = []
        seg_rewards: list[float] = []

        for _ in range(args.max_steps):
            with torch.no_grad():
                q_values = q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            a = sample_action_from_q(q_values, rng, temperature=args.pref_temperature)

            s2, r, done = env.step(ACTIONS[a], render=False)
            seg_states.append(np.asarray(s, dtype=np.float32))
            seg_actions.append(a)
            seg_rewards.append(float(r))

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

            s = s2
            if bool(done):
                break

    return segments


def build_preference_pairs(segments: list[Segment], pair_count: int, seed: int):
    if len(segments) < 2:
        raise RuntimeError("Not enough segments collected to build preference pairs.")

    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(pair_count):
        i, j = rng.choice(len(segments), size=2, replace=False)
        s1, s2 = segments[int(i)], segments[int(j)]
        label = 1.0 if s1.score >= s2.score else 0.0
        pairs.append((s1, s2, label))
    return pairs


def train_reward_model(args, reward_model: RewardModel, pairs) -> None:
    reward_model.train()
    optimizer = optim.Adam(reward_model.parameters(), lr=args.reward_lr)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(args.reward_epochs):
        random.shuffle(pairs)
        epoch_loss = 0.0

        for start in range(0, len(pairs), args.reward_batch_size):
            batch = pairs[start : start + args.reward_batch_size]
            if not batch:
                continue

            logits_list = []
            labels_list = []

            for seg_a, seg_b, label in batch:
                sa = torch.tensor(seg_a.states, dtype=torch.float32)
                aa = torch.tensor(seg_a.actions, dtype=torch.int64)
                sb = torch.tensor(seg_b.states, dtype=torch.float32)
                ab = torch.tensor(seg_b.actions, dtype=torch.int64)

                score_a = reward_model(sa, aa).sum()
                score_b = reward_model(sb, ab).sum()
                logits_list.append(score_a - score_b)
                labels_list.append(label)

            logits = torch.stack(logits_list)
            labels = torch.tensor(labels_list, dtype=torch.float32)
            loss = bce(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        print(f"[reward model] epoch={epoch + 1}/{args.reward_epochs} loss={epoch_loss:.4f}")


def finetune_ddqn_with_reward_model(args, q: DQN, tgt: DQN, reward_model: RewardModel) -> None:
    OBELIX = import_obelix(args.obelix_py)
    rng = np.random.default_rng(args.seed + 1000)
    replay = Replay(args.replay)

    opt = optim.Adam(q.parameters(), lr=args.rl_lr)
    reward_model.eval()
    q.train()
    tgt.eval()

    steps = 0

    def eps_by_step(t: int) -> float:
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    for ep in range(args.rl_episodes):
        env_seed = args.seed + 2000 + ep
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            arena_size=args.arena_size,
            max_steps=args.max_steps,
            wall_obstacles=args.wall_obstacles,
            difficulty=args.difficulty,
            box_speed=args.box_speed,
            seed=env_seed,
        )
        s = env.reset(seed=env_seed)

        ep_env_ret = 0.0
        ep_train_ret = 0.0

        for _ in range(args.max_steps):
            with torch.no_grad():
                qs = q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()

            eps = eps_by_step(steps)
            if rng.random() < eps:
                a = int(rng.integers(ACTION_DIM))
            else:
                a = int(np.argmax(qs))

            s2, env_r, done = env.step(ACTIONS[a], render=False)
            env_r = float(env_r)
            ep_env_ret += env_r

            with torch.no_grad():
                r_hat = reward_model(
                    torch.tensor(s, dtype=torch.float32).unsqueeze(0),
                    torch.tensor([a], dtype=torch.int64),
                ).item()

            train_r = args.reward_mix * env_r + (1.0 - args.reward_mix) * float(r_hat)
            ep_train_ret += train_r

            replay.add(Transition(s=s, a=a, r=train_r, s2=s2, done=bool(done)))
            s = s2
            steps += 1

            if len(replay) >= max(args.warmup, args.batch):
                sb, ab, rb, s2b, db = replay.sample(args.batch)
                sb_t = torch.tensor(sb)
                ab_t = torch.tensor(ab)
                rb_t = torch.tensor(rb)
                s2b_t = torch.tensor(s2b)
                db_t = torch.tensor(db)

                with torch.no_grad():
                    next_q = q(s2b_t)
                    next_a = torch.argmax(next_q, dim=1)
                    next_q_tgt = tgt(s2b_t)
                    next_val = next_q_tgt.gather(1, next_a.unsqueeze(1)).squeeze(1)
                    y = rb_t + args.gamma * (1.0 - db_t) * next_val

                pred = q(sb_t).gather(1, ab_t.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), args.grad_clip)
                opt.step()

                if steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if bool(done):
                break

        if (ep + 1) % args.log_every == 0:
            print(
                f"[rlhf ddqn] episode={ep + 1}/{args.rl_episodes} "
                f"env_return={ep_env_ret:.1f} train_return={ep_train_ret:.1f} eps={eps_by_step(steps):.3f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, default="./obelix_change.py")
    ap.add_argument("--base_weights", type=str, default="weights.pth")
    ap.add_argument("--out", type=str, default="weights_rlhf.pth")
    ap.add_argument("--reward_model_out", type=str, default="reward_model_ddqn.pth")

    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=1000)

    ap.add_argument("--pref_episodes", type=int, default=80)
    ap.add_argument("--segment_len", type=int, default=25)
    ap.add_argument("--pair_count", type=int, default=2000)
    ap.add_argument("--pref_temperature", type=float, default=1.0)

    ap.add_argument("--reward_epochs", type=int, default=8)
    ap.add_argument("--reward_batch_size", type=int, default=32)
    ap.add_argument("--reward_lr", type=float, default=1e-3)

    ap.add_argument("--rl_episodes", type=int, default=300)
    ap.add_argument("--rl_lr", type=float, default=5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--value_coef", type=float, default=0.5)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--reward_mix", type=float, default=0.2)

    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--replay", type=int, default=200000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--target_sync", type=int, default=1000)
    ap.add_argument("--eps_start", type=float, default=0.2)
    ap.add_argument("--eps_end", type=float, default=0.02)
    ap.add_argument("--eps_decay_steps", type=int, default=400000)

    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    q = DQN()
    if os.path.exists(args.base_weights):
        q.load_state_dict(torch.load(args.base_weights, map_location="cpu"), strict=False)
    else:
        print(f"Warning: base weights not found at {args.base_weights}; starting from random DQN.")

    tgt = DQN()
    tgt.load_state_dict(q.state_dict())

    print("Collecting preference segments from DDQN policy...")
    segments = collect_segments(args, q)
    print(f"Collected segments: {len(segments)}")

    print("Building preference pairs...")
    pairs = build_preference_pairs(segments, args.pair_count, seed=args.seed + 1)
    print(f"Preference pairs: {len(pairs)}")

    reward_model = RewardModel()
    print("Training reward model...")
    train_reward_model(args, reward_model, pairs)
    torch.save(reward_model.state_dict(), args.reward_model_out)
    print(f"Saved reward model: {args.reward_model_out}")

    print("Fine-tuning DDQN with reward model...")
    finetune_ddqn_with_reward_model(args, q, tgt, reward_model)
    torch.save(q.state_dict(), args.out)
    print(f"Saved RLHF DDQN policy: {args.out}")


if __name__ == "__main__":
    main()
