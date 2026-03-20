import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

class ActorCritic(nn.Module):
    def __init__(self, in_dim=18, n_actions=5, hidden_dim=64):
        super().__init__()
        # Shared feature extractor
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU()
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
    def __init__(self, state_dim=18, action_dim=5, lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, K_epochs=4, eps_clip=0.2):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr_actor)

        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        self.hidden_dim = 64
        self.reset_hidden()

    def reset_hidden(self):
        self.hidden = (torch.zeros(1, 1, self.hidden_dim),
                       torch.zeros(1, 1, self.hidden_dim))

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).view(1, 1, -1)
            action_probs, state_value, self.hidden = self.policy_old(state_tensor, self.hidden)
            
            action_probs = action_probs.squeeze(0).squeeze(0)
            state_value = state_value.squeeze(0).squeeze(0)

            dist = Categorical(action_probs)
            action = dist.sample()
            logprob = dist.log_prob(action)
            
        self.buffer.states.append(torch.FloatTensor(state))
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        self.buffer.state_values.append(state_value)
            
        return action.item()

    def evaluate(self, states, actions, is_terminals):
        episodes_states = []
        episodes_actions = []
        curr_s = []
        curr_a = []
        
        # Split flat buffer into sequences matching episodes
        for s, a, done in zip(states, actions, is_terminals):
            curr_s.append(s)
            curr_a.append(a)
            if done:
                episodes_states.append(torch.stack(curr_s))
                episodes_actions.append(torch.stack(curr_a))
                curr_s = []
                curr_a = []
        if curr_s: # handle incomplete end
            episodes_states.append(torch.stack(curr_s))
            episodes_actions.append(torch.stack(curr_a))

        logprobs_all = []
        state_values_all = []
        dist_entropy_all = []

        for ep_s, ep_a in zip(episodes_states, episodes_actions):
            # Process sequence as a batch of 1
            ep_s = ep_s.unsqueeze(0) 
            
            # Initial hidden state is zero at start of each episode sequence
            hidden = (torch.zeros(1, 1, self.hidden_dim).to(ep_s.device), 
                      torch.zeros(1, 1, self.hidden_dim).to(ep_s.device))
            
            probs, values, _ = self.policy(ep_s, hidden)
            
            # Remove batch dims
            probs = probs.squeeze(0)
            values = values.squeeze(0).squeeze(-1)
            
            dist = Categorical(probs)
            logprobs = dist.log_prob(ep_a)
            entropy = dist.entropy()
            
            logprobs_all.append(logprobs)
            state_values_all.append(values)
            dist_entropy_all.append(entropy)
            
        return torch.cat(logprobs_all), torch.cat(state_values_all), torch.cat(dist_entropy_all)

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = self.buffer.states
        old_actions = self.buffer.actions
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach()

        # Calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values (uses RNN sequence correctly)
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions, self.buffer.is_terminals)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, default="./obelix.py")
    ap.add_argument("--out", type=str, default="ppo_weights.pth")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--update_timestep", type=int, default=2000)
    ap.add_argument("--difficulty", type=int, default=0)
    ap.add_argument("--wall_obstacles", action="store_true")

    ap.add_argument("--box_speed", type=int, default=2)
    ap.add_argument("--scaling_factor", type=int, default=5)
    ap.add_argument("--arena_size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)

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

    ppo_agent = PPOAgent()
    
    # Load previously saved weights if the file exists
    if os.path.exists(args.out):
        print(f"Loading previous weights from {args.out}...")
        ppo_agent.policy.load_state_dict(torch.load(args.out, map_location="cpu"))
        ppo_agent.policy_old.load_state_dict(ppo_agent.policy.state_dict())

    time_step = 0

    print("Training PPO agent...")

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        ppo_agent.reset_hidden()
        ep_reward = 0
        done = False

        while not done:
            action_idx = ppo_agent.select_action(state)
            next_state, reward, done_info = env.step(ACTIONS[action_idx])
            done = bool(done_info)
            
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            state = next_state
            ep_reward += reward
            time_step += 1

            if time_step % args.update_timestep == 0:
                ppo_agent.update()

        print(f"Episode: {ep}, Reward: {ep_reward:.2f}")
        
        # Save weights every episode to protect against interruptions
        torch.save(ppo_agent.policy.state_dict(), args.out)

    print(f"Training completed. Final weights saved to {args.out}")

if __name__ == "__main__":
    main()
