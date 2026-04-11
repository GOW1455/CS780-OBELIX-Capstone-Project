import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

def fixed_explore_action(step: int) -> tuple[int, int]:
    """Return the scripted exploration action and the next step counter.

    The sequence is 2 left turns, then 30 forward moves, then repeat while the
    observation remains empty.
    """
    if step < 2:
        return 0, step + 1  # L45
    if step < 32:
        return 2, step + 1  # FW
    return 0, 1

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

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = []
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.state_values.clear()

class PPOAgent:
    def __init__(self, state_dim=18, action_dim=5, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=4, eps_clip=0.5,
                 epsilon_start=0.50, epsilon_end=0.02, epsilon_decay=0.9995):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.action_dim = action_dim

        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr_actor)

        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        self.hidden_dim = 128
        self.reset_hidden()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def reset_hidden(self):
        self.hidden = (torch.zeros(1, 1, self.hidden_dim).to(self.device),
                       torch.zeros(1, 1, self.hidden_dim).to(self.device))
        
        # --- REPETITION BAN: Track actions per episode ---
        self.last_action = None
        self.consecutive_count = 0
        self.explore_step = 0

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).view(1, 1, -1).to(self.device)
            action_probs, state_value, self.hidden = self.policy_old(state_tensor, self.hidden)

            # DETACH HIDDEN STATE
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

            action_probs = action_probs.squeeze(0).squeeze(0)
            state_value = state_value.squeeze(0).squeeze(0)

        if np.all(state == 0):
            action_idx, self.explore_step = fixed_explore_action(self.explore_step)
            dist = Categorical(action_probs)
            action = torch.tensor(action_idx, device=self.device)
            logprob = dist.log_prob(action)

            self.buffer.states.append(torch.FloatTensor(state).to(self.device))
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(logprob)
            self.buffer.state_values.append(state_value)
            return action_idx

        self.explore_step = 0

        # ==========================================
        # ACTION MASKING: The Repetition Ban
        # ==========================================
        mask = torch.ones(5).to(self.device)

        # If the same action was taken 5 times in a row, ban it for this step
        if self.last_action is not None and self.consecutive_count >= 5:
            mask[self.last_action] = 0.0

        # Apply the mask
        action_probs = action_probs * mask

        # Re-normalize probabilities so they sum to 1.0
        if action_probs.sum() > 0:
            action_probs = action_probs / action_probs.sum()
        else:
            # Failsafe uniform distribution
            action_probs = torch.ones(5).to(self.device) / 5.0
        # ==========================================

        # Epsilon exploration: occasionally force a random valid action.
        if random.random() < self.epsilon:
            valid_actions = torch.nonzero(mask > 0.0, as_tuple=False).squeeze(-1)
            if valid_actions.numel() == 0:
                valid_actions = torch.arange(self.action_dim, device=self.device)
            rand_idx = torch.randint(valid_actions.numel(), (1,), device=self.device)
            action = valid_actions[rand_idx].squeeze(0)
            dist = Categorical(action_probs)
            logprob = dist.log_prob(action)
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            logprob = dist.log_prob(action)

        action_idx = action.item()

        # --- REPETITION BAN: UPDATE TRACKING COUNTERS ---
        if action_idx == self.last_action:
            self.consecutive_count += 1
        else:
            self.last_action = action_idx
            self.consecutive_count = 1

        self.buffer.states.append(torch.FloatTensor(state).to(self.device))
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        self.buffer.state_values.append(state_value)

        return action_idx

    def evaluate(self, states, actions, is_terminals):
        episodes_states = []
        episodes_actions = []
        curr_s = []
        curr_a = []
        
        for s, a, done in zip(states, actions, is_terminals):
            curr_s.append(s)
            curr_a.append(a)
            if done:
                episodes_states.append(torch.stack(curr_s))
                episodes_actions.append(torch.stack(curr_a))
                curr_s = []
                curr_a = []
        
        if curr_s: 
            episodes_states.append(torch.stack(curr_s))
            episodes_actions.append(torch.stack(curr_a))

        logprobs_all = []
        state_values_all = []
        dist_entropy_all = []

        seq_chunk_size = 64 

        for ep_s, ep_a in zip(episodes_states, episodes_actions):
            hidden = (torch.zeros(1, 1, self.hidden_dim).to(self.device), 
                      torch.zeros(1, 1, self.hidden_dim).to(self.device))
            
            for i in range(0, len(ep_s), seq_chunk_size):
                chunk_s = ep_s[i:i + seq_chunk_size].unsqueeze(0) 
                chunk_a = ep_a[i:i + seq_chunk_size]
                
                probs, values, hidden = self.policy(chunk_s, hidden)
                hidden = (hidden[0].detach(), hidden[1].detach())
                
                probs = probs.squeeze(0)
                values = values.squeeze(0).squeeze(-1)
                
                dist = Categorical(probs)
                logprobs = dist.log_prob(chunk_a)
                entropy = dist.entropy()
                
                logprobs_all.append(logprobs)
                state_values_all.append(values)
                dist_entropy_all.append(entropy)
                
        return torch.cat(logprobs_all), torch.cat(state_values_all), torch.cat(dist_entropy_all)

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = self.buffer.states
        old_actions = self.buffer.actions
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach()

        advantages = rewards.detach() - old_state_values.detach()
        
        # ADVANTAGE NORMALIZATION
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions, self.buffer.is_terminals)

            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.1 * dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, default="./obelix_change.py")
    ap.add_argument("--out", type=str, default="ppo_weights.pth")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--update_timestep", type=int, default=2000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")

    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epsilon_start", type=float, default=0.8)
    ap.add_argument("--epsilon_end", type=float, default=0.02)
    ap.add_argument("--epsilon_decay", type=float, default=0.9995)

    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)
    env = OBELIX(
        difficulty=args.difficulty,
        wall_obstacles=args.wall_obstacles,
        scaling_factor=args.scaling_factor,
        box_speed=args.box_speed,
        arena_size=args.arena_size,
        seed=args.seed
    )

    ppo_agent = PPOAgent(
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
    )
    
    if os.path.exists(args.out):
        print(f"Loading previous weights from {args.out}...")
        ppo_agent.policy.load_state_dict(torch.load(args.out, map_location=ppo_agent.device))
        ppo_agent.policy_old.load_state_dict(ppo_agent.policy.state_dict())

    time_step = 0
    print(f"Training PPO agent on device: {ppo_agent.device}...")

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        ppo_agent.reset_hidden()
        ep_reward = 0
        done = False

        while not done:
            action_idx = ppo_agent.select_action(state)
            next_state, reward, done_info = env.step(ACTIONS[action_idx], render=True)
            done = bool(done_info)
            
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            state = next_state
            ep_reward += reward
            time_step += 1

            if time_step >= args.update_timestep and done:
                ppo_agent.update()
                time_step = 0 

        ppo_agent.decay_epsilon()
        print(f"Episode: {ep}, Reward: {ep_reward:.2f}, Epsilon: {ppo_agent.epsilon:.4f}")
        
        torch.save(ppo_agent.policy.state_dict(), args.out)

    print(f"Training completed. Final weights saved to {args.out}")

if __name__ == "__main__":
    main()