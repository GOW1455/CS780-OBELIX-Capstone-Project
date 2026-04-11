from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

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


_model: Optional[ActorCritic] = None


def _load_once() -> None:
    global _model
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    preferred = os.path.join(here, "ac_weights.pth")
    fallback = os.path.join(here, "weights.pth")

    if os.path.exists(preferred):
        wpath = preferred
    elif os.path.exists(fallback):
        wpath = fallback
    else:
        raise FileNotFoundError("ac_weights.pth or weights.pth not found next to agent_actor_critic.py")

    m = ActorCritic()
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=False)
    m.eval()
    _model = m


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_once()
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits, _ = _model(x)
    probs = torch.softmax(logits.squeeze(0), dim=-1).cpu().numpy()
    probs_sum = float(np.sum(probs))
    if not np.isfinite(probs_sum) or probs_sum <= 0.0:
        probs = np.ones(len(ACTIONS), dtype=np.float32) / len(ACTIONS)
    else:
        probs = probs / probs_sum
    return ACTIONS[int(rng.choice(len(ACTIONS), p=probs))]
