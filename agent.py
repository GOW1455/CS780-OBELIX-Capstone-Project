"""OBELIX evaluation agent with probabilistic Q-action selection.

Behavior:
- Sample actions from a probability distribution derived from Q-values.

Place weights.pth next to this file for evaluation.
"""

from __future__ import annotations

from typing import Optional
import os

import numpy as np
import torch
import torch.nn as nn
from policy import policy_function

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]


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


_model: Optional[DQN] = None


def _load_once() -> None:
    global _model
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent_explore.py. Train offline and place weights.pth there."
        )

    model = DQN()
    state_dict = torch.load(wpath, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    _model = model


def _sample_action_from_qs(qs: np.ndarray, rng: np.random.Generator) -> int:
    qs = np.asarray(qs, dtype=np.float32)
    shifted = (qs - np.max(qs)) * 5.0
    probs = np.exp(shifted)
    probs_sum = float(np.sum(probs))
    if not np.isfinite(probs_sum) or probs_sum <= 0.0:
        probs = np.ones(len(ACTIONS), dtype=np.float32) / len(ACTIONS)
    else:
        probs = probs / probs_sum
    return int(rng.choice(len(ACTIONS), p=probs))


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    obs = np.asarray(obs)
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    qs = _model(x).squeeze(0).cpu().numpy()
    action_idx = _sample_action_from_qs(qs, rng)
    action = policy_function(obs, rng)
    return action
