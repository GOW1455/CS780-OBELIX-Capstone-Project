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
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    q = _model(x).squeeze(0).cpu().numpy()
    
    # PRIORITY 1: Wall Avoidance (Interleaved R45 and FW)
    if obs[17] == 1 or recovery_steps > 0:
        if obs[17] == 1 and recovery_steps == 0:
            # Sequence: R45 -> FW -> R45 -> FW -> 10x FW (Total 14 steps)
            recovery_steps = 14  
        
        recovery_steps -= 1
        step_counter = 0  # Reset exploration loop
        
        # Steps 13 and 11: Rotate Right 45
        if recovery_steps in [13, 12, 11]:
            return "L45"
        # All other recovery steps (including after each turn): Move Forward
        return "FW"
    
    # PRIORITY 2: Box Targeting
    if obs[16] == 1:
        return "FW"
    
    # Interleaving Forward after directional turns
    if any(obs[0:4]):  # Box is on the left
        # Logic: If step is even, turn; if odd, go FW
        step_counter += 1
        return "R45"
    if any(obs[12:16]): # Box is on the right
        step_counter += 1
        return "L45"
    
    if any(obs[4:12]): # Box is ahead
        return "FW"
    
    # PRIORITY 3: Default Exploration (Alternating L22 and FW)
    # This will take 5 turns of L22, each followed by a FW step
    if step_counter < 1:
        action = "R45"
    else:
        action = "FW"
    
    # Increased cycle to allow for the 10 forward steps after the turns
    step_counter = (step_counter + 1) % 70 
    return action
