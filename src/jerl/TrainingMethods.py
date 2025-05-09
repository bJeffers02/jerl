import time
import torch
import torch.nn.functional as F
import numpy as np


from collections import namedtuple
SavedAction = namedtuple('SavedAction', ['action', 'prob', 'value'])
eps = np.finfo(np.float32).eps.item()


class _TrainingMethod:
    def __init__(self, optimizer, device=torch.device('cpu'), scheduler=None):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"'optimizer' must be a torch.optim.Optimizer, got {type(optimizer).__name__}")
        if device is not None and not isinstance(device, torch.device):
            raise TypeError(f"'device' must be a torch.device, got {type(device).__name__}")
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
            raise TypeError(f"'scheduler' must be a torch.optim.lr_scheduler._LRScheduler, got {type(scheduler).__name__}")
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.params = [
        p for group in self.optimizer.param_groups
        for p in group['params'] if p.grad is not None
        ]

        self.episode_actions = []
        self.episode_rewards = []


    def save_action(self, action, prob: torch.Tensor, value: torch.Tensor):
        self.episode_actions.append(SavedAction(action, prob, value))


    def save_reward(self, reward):
        self.episode_rewards.append(reward)


class A2C(_TrainingMethod):
    def __init__(self, device, optimizer, scheduler, gamma=0.9, gae_lambda=0.95, initial_entropy_coef=0.1, min_entropy_coef=0.001, **kwargs):
        super(A2C, self).__init__(device, optimizer, scheduler)

        if kwargs:
            for key in kwargs:
                print(f"Unexpected argument: '{key}' with value '{kwargs[key]}'")

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.initial_entropy_coef = initial_entropy_coef
        self.min_entropy_coef = min_entropy_coef

        if not isinstance(self.gamma, float):
            raise TypeError(f"'gamma' must be a float, got {type(self.gamma).__name__}")
        if not isinstance(self.gae_lambda, float):
            raise TypeError(f"'gae_lambda' must be a float, got {type(self.gae_lambda).__name__}")
        if not isinstance(self.initial_entropy_coef, float):
            raise TypeError(f"'initial_entropy_coef' must be a float, got {type(self.initial_entropy_coef).__name__}")
        if not isinstance(self.min_entropy_coef, float):
            raise TypeError(f"'min_entropy_coef' must be a float, got {type(self.min_entropy_coef).__name__}")
        

    def train(self):
        print("Training on Episode Data...")
        start_time = time.perf_counter()

        rewards = torch.tensor(self.episode_rewards, device=self.device)
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        discounted_sum = 0

        # Initialize 'values' here (ensure it's the output from your critic model)
        values = torch.stack([a.value for a in self.episode_actions]).squeeze()

        # Calculate returns and advantages (with GAE for smoothing)
        for t in reversed(range(len(rewards))):
            discounted_sum = rewards[t] + self.gamma * discounted_sum
            returns[t] = discounted_sum
            
            if t < len(rewards) - 1:
                # GAE calculation
                delta_t = rewards[t] + self.gamma * values[t+1] - values[t]
                advantages[t] = delta_t + self.gamma * self.gae_lambda * advantages[t+1]
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
        entropy_coef = max(self.min_entropy_coef, self.initial_entropy_coef * uncertainty)

        self.optimizer.zero_grad()

        # Total loss
        loss = actor_loss + critic_loss - entropy_coef * entropy
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()

        self.episode_actions.clear()
        self.episode_rewards.clear()

        duration = time.perf_counter() - start_time
        print(f"Episode Training Complete in {duration:.2f}s.")


# class SAC(_TrainingMethod):
#     def __init__(self, device, optimizer, scheduler, **kwargs):
#         super(SAC, self).__init__(device, optimizer, scheduler)


#     def train(self):
#         print("Error: SAC training method is not yet implemented.")
#         raise NotImplementedError("SAC training method is under development.")

# class PPO(_TrainingMethod):
#     def __init__(self, device, optimizer, scheduler, **kwargs):
#         super(PPO, self).__init__(device, optimizer, scheduler)


#     def train(self):
#         print("Error: PPO training method is not yet implemented.")
#         raise NotImplementedError("PPO training method is under development.")