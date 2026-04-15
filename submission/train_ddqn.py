"""Offline trainer: Double DQN + replay buffer (CPU) for OBELIX.

Run locally to create weights.pth, then submit agent.py + weights.pth.

Example:
  python train_dqn.py --obelix_py ./obelix.py --out weights.pth --episodes 2000 --difficulty 0 --wall_obstacles

                        ALGORITHM: DOUBLE DEEP Q-NETWORK (DDQN)


Double DQN is one of the most widely used and reliable improvements over the original Deep Q-Network (DQN).

Main problems it solves:
Vanilla DQN often overestimates true action values.
This happens because the same network is used twice:
   1. to pick the best-looking action in the next state (max)
   2. to evaluate how good that action actually is

When Q-values are noisy (which they almost always are early in training),
this double usage creates optimistic bias → the agent thinks some
actions are much better than they really are → leads to unstable learning.

Double DQN solution:
Split the responsibilities:
• Use the online / main Q-network  to SELECT which action looks best
• Use the target Q-network to EVALUATE (give the actual value)

So instead of:

    target = r + γ × max_a Q_target(s', a)

We do:

    target = r + γ × Q_target( s',   argmax_a Q_online(s', a)   )

This small change dramatically reduces overestimation and makes learning
much more stable — especially in environments with large action spaces
or noisy rewards.

For More Details please refer to https://arxiv.org/pdf/1509.06461 .


"""

from __future__ import annotations
import argparse, random
import csv
from collections import deque
from dataclasses import dataclass
import os
from typing import Deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
RANDOM_ACTION_PROBS = np.array([0.1, 0.15, 0.5, 0.15, 0.1], dtype=np.float32)


def sample_action_from_qs(qs: np.ndarray, rng: np.random.Generator) -> int:
    qs = np.asarray(qs, dtype=np.float32)
    shifted = (qs - np.max(qs)) * 5.0  # for numerical stability
    probs = np.exp(shifted)
    probs_sum = float(np.sum(probs))
    if not np.isfinite(probs_sum) or probs_sum <= 0.0:
        probs = np.ones(len(ACTIONS), dtype=np.float32) / len(ACTIONS)
    else:
        probs = probs / probs_sum
    return int(rng.choice(len(ACTIONS), p=probs))


def sample_action_from_fixed_probs(rng: np.random.Generator) -> int:
    probs = RANDOM_ACTION_PROBS
    probs_sum = float(np.sum(probs))
    if not np.isfinite(probs_sum) or probs_sum <= 0.0:
        probs = np.ones(len(ACTIONS), dtype=np.float32) / len(ACTIONS)
    else:
        probs = probs / probs_sum
    return int(rng.choice(len(ACTIONS), p=probs))


def select_action(qs: np.ndarray, rng: np.random.Generator, epsilon: float) -> int:
    if rng.random() < epsilon:
        return sample_action_from_fixed_probs(rng)
    return sample_action_from_qs(qs, rng)

def fixed_explore_action(step: int) -> tuple[int, int]:
    if step < 6:
        return 1, (step + 1)%50
    if step < 50:
        return 2, (step + 1)%50
    return 0, 1

class DQN(nn.Module):
    def __init__(self, in_dim=18, n_actions=5):
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
    def forward(self, x):
        return self.net(x)

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
    def __len__(self): return len(self.buf)

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, default="./obelix.py")
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--rewards_csv", type=str, default="training_rewards.csv")
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--replay", type=int, default=200000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--target_sync", type=int, default=1000)
    ap.add_argument("--eps_start", type=float, default=0.3)
    ap.add_argument("--eps_end", type=float, default=0.01)
    ap.add_argument("--eps_decay_steps", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    OBELIX = import_obelix(args.obelix_py)
    print("Training Double DQN agent...")


    q = DQN()
    if os.path.exists(args.out):
        q.load_state_dict(torch.load(args.out, map_location="cpu"))
    tgt = DQN()
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt = optim.Adam(q.parameters(), lr=args.lr)
    replay = Replay(args.replay)
    steps = 0

    def eps_by_step(t):
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    csv_exists = os.path.exists(args.rewards_csv)
    with open(args.rewards_csv, "a", newline="", encoding="utf-8") as rewards_fp:
        csv_writer = csv.writer(rewards_fp)
        if (not csv_exists) or os.path.getsize(args.rewards_csv) == 0:
            csv_writer.writerow(["episode", "return", "epsilon", "replay_size", "steps"])

        for ep in range(args.episodes):
            print(f"Episode {ep+1}/{args.episodes} starting...")
            env = OBELIX(
                scaling_factor=args.scaling_factor,
                arena_size=args.arena_size,
                max_steps=args.max_steps,
                wall_obstacles=args.wall_obstacles,
                difficulty=args.difficulty,
                box_speed=args.box_speed,
                seed=args.seed + ep,
            )
            s = env.reset(seed=args.seed + ep*42)
            ep_ret = 0.0
            explore_step = 0

            for _ in range(args.max_steps):
                # if np.all(s == 0):
                #     a, explore_step = fixed_explore_action(explore_step)
                # else:
                #     explore_step = 0
                #     with torch.no_grad():
                #         qs = q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                #     a = select_action(qs, rng, eps_by_step(steps))
                with torch.no_grad():
                    qs = q(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                a = select_action(qs, rng, eps_by_step(steps))
                
                s2, r, done = env.step(ACTIONS[a], render=True)
                ep_ret += float(r)
                replay.add(Transition(s=s, a=a, r=float(r), s2=s2, done=bool(done)))
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
                    nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                    opt.step()

                    if steps % args.target_sync == 0:
                        tgt.load_state_dict(q.state_dict())

                if done:
                    break

            torch.save(q.state_dict(), args.out)
            ep_eps = eps_by_step(steps)
            csv_writer.writerow([ep + 1, ep_ret, ep_eps, len(replay), steps])
            rewards_fp.flush()

            if (ep + 1) % 1 == 0:
                print(f"Episode {ep+1}/{args.episodes} return={ep_ret:.1f} eps={ep_eps:.3f} replay={len(replay)}")

    torch.save(q.state_dict(), args.out)
    print("Saved:", args.out)
    print("Rewards CSV:", args.rewards_csv)

if __name__ == "__main__":
    main()
