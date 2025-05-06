import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import namedtuple
SavedAction = namedtuple('SavedAction', ['action', 'prob', 'value'])

class ActorCriticNN(nn.Module):
    def __init__(self, network_dims, device):
        print("Creating Model...")

        self.device = device

        super(ActorCriticNN, self).__init__()
        num_layers = len(network_dims)
        self.layers = nn.ModuleList()
        for i in range(num_layers-2):
            self.layers.append(nn.Linear(network_dims[i], network_dims[i+1]))
        self.actor_layer = nn.Linear(network_dims[num_layers-2], network_dims[num_layers-1])
        self.critic_layer = nn.Linear(network_dims[num_layers-2], 1)
        self.episode_actions = []
        self.episode_rewards = []

        self.to(self.device)
        print(f"Model Created on \"{self.device}\".")

    
    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        action = F.softmax(self.actor_layer(x), dim=-1)
        state_value = self.critic_layer(x)
        return action, state_value
    
    def save_action(self, action, prob: torch.Tensor, value: torch.Tensor):
        self.episode_actions.append(SavedAction(action, prob, value))
    
    def train(self, optimizer, scheduler, cfg):
        print("Training on Episode Data...")
        start_time = time.perf_counter()

        eps = np.finfo(np.float32).eps.item()

        rewards = torch.tensor(self.episode_rewards, device=self.device)
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        discounted_sum = 0
        lamda = 0.95  # GAE lamda (can tune between 0 and 1)

        # Initialize 'values' here (ensure it's the output from your critic model)
        values = torch.stack([a.value for a in self.episode_actions]).squeeze()

        # Calculate returns and advantages (with GAE for smoothing)
        for t in reversed(range(len(rewards))):
            discounted_sum = rewards[t] + cfg.reward_gamma * discounted_sum
            returns[t] = discounted_sum
            
            if t < len(rewards) - 1:
                # GAE calculation
                delta_t = rewards[t] + cfg.reward_gamma * values[t+1] - values[t]
                advantages[t] = delta_t + cfg.reward_gamma * lamda * advantages[t+1]
            else:
                advantages[t] = rewards[t] - values[t]

        returns = (returns - returns.mean()) / (returns.std() + eps)

        actions = torch.stack([a.action for a in self.episode_actions]) 
        probs = torch.stack([a.prob for a in self.episode_actions])
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)) + eps).squeeze()

        # Calculate the advantages
        advantages = returns - values.detach()

        # Actor-Critic Loss
        actor_loss = (-log_probs * advantages).sum()
        critic_loss = F.smooth_l1_loss(values, returns, reduction='sum')

        # Entropy term for exploration
        entropy = -(probs * probs.log()).sum(dim=1).mean()
        uncertainty = torch.var(probs, dim=1).mean()  # Variance as exploration measure
        entropy_coef = max(cfg.min_entropy_coef, cfg.initial_entropy_coef * uncertainty)

        avg_reward = rewards.mean().item()
        optimizer.zero_grad()

        # Total loss
        loss = actor_loss + critic_loss - entropy_coef * entropy
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        self.episode_actions.clear()
        self.episode_rewards.clear()

        duration = time.perf_counter() - start_time
        print(f"Episode Training Complete in {duration:.2f}s.")



    # def train(self, optimizer, scheduler, cfg):
    #     print("Training on Episode Data...")
    #     start_time = time.perf_counter()

    #     eps = np.finfo(np.float32).eps.item()

    #     rewards = torch.tensor(self.episode_rewards, device=self.device)
    #     returns = torch.zeros_like(rewards)
    #     discounted_sum = 0
    #     for t in reversed(range(len(rewards))):
    #         discounted_sum = rewards[t] + cfg.reward_gamma * discounted_sum
    #         returns[t] = discounted_sum
    #     returns = (returns - returns.mean()) / (returns.std() + eps)

    #     actions = torch.stack([a.action for a in self.episode_actions]) 
    #     probs = torch.stack([a.prob for a in self.episode_actions])
    #     values = torch.stack([a.value for a in self.episode_actions]).squeeze()
    #     log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)) + eps).squeeze()
    #     advantages = returns - values.detach()

    #     actor_loss = (-log_probs * advantages).sum()
    #     critic_loss = F.smooth_l1_loss(values, returns, reduction='sum')
    #     entropy = -(probs * probs.log()).sum(dim=1).mean()

    #     avg_reward = rewards.mean().item()
    #     decay = np.exp(-cfg.entropy_decay * (avg_reward - 0.25))
    #     entropy_coef = max(cfg.min_entropy_coef, cfg.initial_entropy_coef * decay)

    #     optimizer.zero_grad()
    #     loss = actor_loss + critic_loss - entropy_coef * entropy
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
    #     optimizer.step()
    #     scheduler.step()

    #     self.episode_actions.clear()
    #     self.episode_rewards.clear()

    #     duration = time.perf_counter() - start_time
    #     print(f"Episode Training Complete in {duration:.2f}s.")