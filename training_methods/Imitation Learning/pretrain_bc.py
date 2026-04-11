from __future__ import annotations

import argparse
import os
import pickle
import random
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


class DQN(nn.Module):
    def __init__(self, in_dim: int = 18, n_actions: int = 5):
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


def _to_action_index(a) -> int:
    if isinstance(a, str):
        if a not in ACTION_TO_IDX:
            raise ValueError(f"Unknown action string: {a}")
        return ACTION_TO_IDX[a]
    i = int(a)
    if i < 0 or i >= len(ACTIONS):
        raise ValueError(f"Action index out of range: {i}")
    return i


def _iter_samples(human_data) -> Iterable[tuple[np.ndarray, int]]:
    # Supported formats:
    # 1) list of (episode_states, episode_actions)
    # 2) dict with keys: states, actions
    # 3) list of dicts with keys: state/obs, action
    if isinstance(human_data, dict) and "states" in human_data and "actions" in human_data:
        states = human_data["states"]
        actions = human_data["actions"]
        for s, a in zip(states, actions):
            yield np.asarray(s, dtype=np.float32), _to_action_index(a)
        return

    for item in human_data:
        if isinstance(item, dict):
            if "state" in item and "action" in item:
                yield np.asarray(item["state"], dtype=np.float32), _to_action_index(item["action"])
                continue
            if "obs" in item and "action" in item:
                yield np.asarray(item["obs"], dtype=np.float32), _to_action_index(item["action"])
                continue
            raise ValueError("Dict sample must contain (state, action) or (obs, action).")

        if isinstance(item, (tuple, list)) and len(item) == 2:
            ep_states, ep_actions = item
            for s, a in zip(ep_states, ep_actions):
                yield np.asarray(s, dtype=np.float32), _to_action_index(a)
            continue

        raise ValueError("Unsupported human_data entry format.")


def train_behavioral_cloning() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="human_data.pkl")
    ap.add_argument("--init_weights", type=str, default="weights.pth")
    ap.add_argument("--out", type=str, default="weights.pth")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.data, "rb") as f:
        human_data = pickle.load(f)

    samples = list(_iter_samples(human_data))
    if not samples:
        raise RuntimeError("No samples found in human_data.")

    states = np.stack([s for s, _ in samples]).astype(np.float32)
    actions = np.asarray([a for _, a in samples], dtype=np.int64)

    model = DQN()
    if os.path.exists(args.init_weights):
        model.load_state_dict(torch.load(args.init_weights, map_location="cpu"), strict=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    n = len(states)
    print(f"Loaded {n} state-action samples from {args.data}")

    model.train()
    for epoch in range(args.epochs):
        idx = np.random.permutation(n)
        epoch_loss = 0.0

        for start in range(0, n, args.batch_size):
            bidx = idx[start : start + args.batch_size]
            sb = torch.tensor(states[bidx], dtype=torch.float32)
            ab = torch.tensor(actions[bidx], dtype=torch.long)

            logits = model(sb)
            loss = criterion(logits, ab)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        print(f"Epoch {epoch + 1}/{args.epochs} loss={epoch_loss:.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"Pretraining complete. Saved DDQN-compatible weights to {args.out}")


if __name__ == "__main__":
    train_behavioral_cloning()
