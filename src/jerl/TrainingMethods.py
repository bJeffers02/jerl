try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is not installed. Please install it manually:\n"
        "CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "Or visit https://pytorch.org/get-started/locally for CUDA options."
    )

import time
import torch.nn.functional as F
import numpy as np


from collections import namedtuple
SavedAction = namedtuple('SavedAction', ['action', 'prob', 'value'])
eps = np.finfo(np.float32).eps.item()


class _TrainingMethod:
    def __init__(self, optimizer, device=torch.device('cpu'), scheduler=None):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"'optimizer' must be a torch.optim.Optimizer, got {type(optimizer).__name__}")
        if not isinstance(device, torch.device):
            raise TypeError(f"'device' must be a torch.device, got {type(device).__name__}")
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            raise TypeError(f"'scheduler' must be a torch.optim.lr_scheduler.LRScheduler, got {type(scheduler).__name__}")
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.params = [
        p for group in self.optimizer.param_groups
        for p in group['params']
        ]

        self.episode_actions = []
        self.episode_rewards = []


    def save_action(self, action, prob: torch.Tensor, value: torch.Tensor):
        if not torch.is_tensor(prob):
            raise TypeError(f"`prob` must be a torch.Tensor, got {type(prob)}")
        target = torch.tensor(1.0, device=prob.device, dtype=prob.dtype)
        if not torch.allclose(prob.sum(), target, atol=1e-3):
            raise ValueError("`prob` must sum to ~1 (invalid action distribution)")
        # if value.dim() != 1:
        #     raise ValueError(f"`value` must be 1D (got shape {value.shape})")
        self.episode_actions.append(SavedAction(action, prob, value))


    def save_reward(self, reward):
        if isinstance(reward, torch.Tensor):
            if reward.numel() != 1:
                raise ValueError(f"Expected scalar tensor, got tensor with shape {reward.shape}")
            reward = reward.item()
        elif not isinstance(reward, (float, int)):
            raise TypeError(f"Reward must be a float, int, or scalar tensor. Got: {type(reward)}")
        
        self.episode_rewards.append(reward)


class A2C(_TrainingMethod):
    def __init__(self, optimizer, device, scheduler,
                gamma=0.9, 
                gae_lambda=0.95, 
                initial_entropy_coef=0.1, 
                min_entropy_coef=0.001, 
                entropy_decay=0.99, 
                max_grad_norm=1.0, 
                **kwargs
                ):
        super(A2C, self).__init__(optimizer, device, scheduler)

        if kwargs:
            for key in kwargs:
                print(f"Unexpected argument: '{key}' with value '{kwargs[key]}'")

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = initial_entropy_coef
        self.min_entropy_coef = min_entropy_coef
        self.entropy_decay = entropy_decay
        self.max_grad_norm = max_grad_norm

        if not isinstance(self.gamma, float):
            raise TypeError(f"'gamma' must be a float, got {type(self.gamma).__name__}")
        if not isinstance(self.gae_lambda, float):
            raise TypeError(f"'gae_lambda' must be a float, got {type(self.gae_lambda).__name__}")
        if not isinstance(self.entropy_coef, float):
            raise TypeError(f"'initial_entropy_coef' must be a float, got {type(self.entropy_coef).__name__}")
        if not isinstance(self.min_entropy_coef, float):
            raise TypeError(f"'min_entropy_coef' must be a float, got {type(self.min_entropy_coef).__name__}")
        if not isinstance(self.entropy_decay, float):
            raise TypeError(f"'entropy_decay' must be a float, got {type(self.entropy_decay).__name__}")
        if not isinstance(self.max_grad_norm, float):
            raise TypeError(f"'max_grad_norm' must be a float, got {type(self.max_grad_norm).__name__}")
        

    def train(self):
        if not self.episode_actions:
            raise RuntimeError("No actions saved. Call `save_action()` first.")
        if not self.episode_rewards:
            raise RuntimeError("No rewards saved. Call `save_reward()` first.")

        print("Training on Episode Data...")
        start_time = time.perf_counter()

        values = torch.stack([a.value for a in self.episode_actions]).squeeze()
        actions = torch.stack([a.action for a in self.episode_actions])
        probs = torch.stack([a.prob for a in self.episode_actions])
        rewards = torch.tensor(self.episode_rewards, device=self.device, dtype=values.dtype)

        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] - values[t] if t < len(rewards)-1 else rewards[t] - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * last_advantage
            last_advantage = advantages[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)
        advantages = torch.clamp(advantages, -10, 10)
        returns = advantages + values.detach()

        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            raise ValueError("NaN/Inf detected in advantages. Check rewards/values.")

        probs = probs.squeeze(1) 
        log_probs = torch.log(probs.gather(1, actions) + eps).squeeze()

        actor_loss = (-log_probs * advantages).sum()
        critic_loss = F.mse_loss(values, returns)

        entropy = -(probs * probs.log()).sum(dim=1).mean()
        if torch.isnan(entropy) or torch.isinf(entropy):
            entropy = torch.tensor(0.0, device=entropy.device)
        self.entropy_coef = max(self.min_entropy_coef, self.entropy_coef * self.entropy_decay)

        self.optimizer.zero_grad()

        loss = actor_loss + critic_loss - self.entropy_coef * entropy
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.max_grad_norm)
        self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()

        self.episode_actions.clear()
        self.episode_rewards.clear()

        duration = time.perf_counter() - start_time
        print(f"Episode Training Complete in {duration:.2f}s.")

        metrics = {
            'total_loss': loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'entropy_coef': self.entropy_coef,
            'mean_advantage': advantages.mean().item(),
            'training_duration': duration
        }

        return metrics


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