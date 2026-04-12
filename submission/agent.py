"""Better DQN-style agent scaffold for OBELIX (CPU).

This agent is *evaluation-only*: it loads pretrained weights from a file
placed next to agent.py inside the submission zip (weights.pth).

The policy samples actions from a probability distribution derived from
the network outputs.

Submission ZIP structure:
  submission.zip
    agent.py
    weights.pth
"""

from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
RANDOM_ACTION_RATE = 0.15
RANDOM_ACTION_PROBS = np.array([0.25, 0.15, 0.3, 0.15, 0.25], dtype=np.float32)
_rap_sum = float(np.sum(RANDOM_ACTION_PROBS))
if np.isfinite(_rap_sum) and _rap_sum > 0.0:
    RANDOM_ACTION_PROBS = RANDOM_ACTION_PROBS / _rap_sum
else:
    RANDOM_ACTION_PROBS = np.ones(len(ACTIONS), dtype=np.float32) / len(ACTIONS)

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


def _sample_action_from_logits(logits: np.ndarray, rng: np.random.Generator) -> int:
    probs = torch.softmax(torch.from_numpy(logits.astype(np.float32)), dim=0).cpu().numpy()
    probs_sum = float(np.sum(probs))
    if not np.isfinite(probs_sum) or probs_sum <= 0.0:
        probs = np.ones(len(ACTIONS), dtype=np.float32) / len(ACTIONS)
    else:
        probs = probs / probs_sum
    return int(rng.choice(len(ACTIONS), p=probs))

def _load_once():
    global _model
    if _model is not None:
        return
    here = os.path.dirname(__file__)
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            "weights.pth not found next to agent.py. Train offline and include it in the submission zip."
        )
    m = DQN()
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=True)
    m.eval()
    _model = m

step_counter = 0
recovery_steps = 0
@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global step_counter, recovery_steps
    _load_once()

    if rng.random() < RANDOM_ACTION_RATE:
        return ACTIONS[int(rng.choice(len(ACTIONS), p=RANDOM_ACTION_PROBS))]

    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    q = _model(x).squeeze(0).cpu().numpy()

    if obs[17] == 1 or recovery_steps > 0:
        if obs[17] == 1 and recovery_steps == 0:
            recovery_steps = 14  

        recovery_steps -= 1
        step_counter = 0
        if recovery_steps in [13, 12, 11]:
            return "L45"
        return "FW"

    if obs[16] == 1:
        return "FW"

    if any(obs[0:4]):
        step_counter += 1
        return "R45"
    if any(obs[12:16]):
        step_counter += 1
        return "L45"

    if any(obs[4:12]):
        return "FW"

    if step_counter < 1:
        action = "R45"
    else:
        action = "FW"

    step_counter = (step_counter + 1) % 70 
    return action
