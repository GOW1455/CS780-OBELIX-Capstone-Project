"""Evaluation agent for PPO.

This agent loads pretrained weights from a file (ppo_weights.pth or weights.pth)
placed next to agent_ppo.py. During evaluation, it acts greedily by picking
the action with the highest probability.
"""

from __future__ import annotations
from typing import List, Optional
import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]

class ActorCritic(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden_dim=128):
        super().__init__()
        # Shared feature extractor
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.Tanh(), # Tanh generally provides more stable gradients for PPO than ReLU
            nn.Linear(256, hidden_dim),
            nn.Tanh()
        )
        
        # LSTM layer for partial observability
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state, hidden):
        # state must be shape: (batch_size, sequence_length, in_dim)
        x = self.fc(state)
        x, hidden = self.lstm(x, hidden)
        
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value, hidden

_model: Optional[ActorCritic] = None
_last_action: Optional[int] = None
_repeat_count: int = 0
_hidden_state = None

_MAX_REPEAT = 2
_CLOSE_PROB_DELTA = 0.05

def _load_once():
    global _model
    if _model is not None:
        return
    here = os.path.dirname(__file__)
    
    # Try ppo_weights.pth first, fallback to weights.pth
    wpath = os.path.join(here, "ppo_weights.pth")
    if not os.path.exists(wpath):
        wpath = os.path.join(here, "weights.pth")
        if not os.path.exists(wpath):
            raise FileNotFoundError(
                "ppo_weights.pth or weights.pth not found next to agent_ppo.py."
            )
            
    m = ActorCritic()
    sd = torch.load(wpath, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    m.load_state_dict(sd, strict=False) # allow strict=False in case critic weights are slightly off
    m.eval()
    _model = m

@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_count, _hidden_state
    _load_once()
    
    # Needs to be float tensor with shape [1, 1, seq_len]
    x = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)
    
    if _hidden_state is None:
        _hidden_state = (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))
    
    # Get action probabilities from the actor
    action_probs, _, _hidden_state = _model(x, _hidden_state)
    probs = action_probs.squeeze().cpu().numpy()
    
    # Act greedily during evaluation
    best = int(np.argmax(probs))

    # Smoothing: if top-2 probabilities are close, avoid flip-flopping
    # if _last_action is not None:
    #     order = np.argsort(-probs)
    #     best_p, second_p = float(probs[order[0]]), float(probs[order[1]])
    #     if (best_p - second_p) < _CLOSE_PROB_DELTA:
    #         if _repeat_count < _MAX_REPEAT:
    #             best = _last_action
    #             _repeat_count += 1
    #         else:
    #             _repeat_count = 0
    #     else:
    #         _repeat_count = 0

    # _last_action = best
    return ACTIONS[best]