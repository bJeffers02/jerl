import time
import torch
import torch.nn.functional as F
import numpy as np

from collections import namedtuple
SavedAction = namedtuple('SavedAction', ['action', 'prob', 'value'])

class _TrainingMethod():
    def __init__(self, device, cfg, optimizer, scheduler):
        self.cfg = cfg
        self.device = device
        self.episode_actions = []
        self.episode_rewards = []
        
        self.optimizer = optimizer
        self.scheduler = scheduler

    def save_action(self, action, prob: torch.Tensor, value: torch.Tensor):
        self.episode_actions.append(SavedAction(action, prob, value))

    def save_reward(self, reward):
        self.episode_rewards.append(reward)

    def train(self):
        raise NotImplementedError("The train method must be overridden by the subclass.")

class A2C(_TrainingMethod):
    def __init__(self, device, cfg, optimizer, scheduler):
        super(A2C, self).__init__(device, cfg, optimizer, scheduler)

    def train(self):
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
            discounted_sum = rewards[t] + self.cfg.reward_gamma * discounted_sum
            returns[t] = discounted_sum
            
            if t < len(rewards) - 1:
                # GAE calculation
                delta_t = rewards[t] + self.cfg.reward_gamma * values[t+1] - values[t]
                advantages[t] = delta_t + self.cfg.reward_gamma * lamda * advantages[t+1]
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
        entropy_coef = max(self.cfg.min_entropy_coef, self.cfg.initial_entropy_coef * uncertainty)

        self.optimizer.zero_grad()

        # Total loss
        loss = actor_loss + critic_loss - entropy_coef * entropy
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.episode_actions.clear()
        self.episode_rewards.clear()

        duration = time.perf_counter() - start_time
        print(f"Episode Training Complete in {duration:.2f}s.")

class SAC(_TrainingMethod):
    def __init__(self, device, cfg, optimizer, scheduler):
        super(SAC, self).__init__(device, cfg, optimizer, scheduler)

    def train(self):
        print("Error: SAC training method is not yet implemented.")
        raise NotImplementedError("SAC training method is under development.")

class PPO(_TrainingMethod):
    def __init__(self, device, cfg, optimizer=None, scheduler=None):
        super(PPO, self).__init__(device, cfg, optimizer, scheduler)

    def train(self):
        print("Error: PPO training method is not yet implemented.")
        raise NotImplementedError("PPO training method is under development.")