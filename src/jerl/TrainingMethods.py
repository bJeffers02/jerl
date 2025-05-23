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
        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        elif isinstance(optimizer, (list, tuple)):
            if len(optimizer) == 1:
                self.optimizer = optimizer[0]
            elif len(optimizer) == 2:
                self.actor_optimizer = optimizer[0]
                self.critic_optimizer = optimizer[1]
            else:
                raise ValueError(f"'optimizer' list must contain 1 or 2 optimizers, got {len(optimizer)}")
            if not all(isinstance(opt, torch.optim.Optimizer) for opt in optimizer):
                bad_types = [type(opt).__name__ for opt in optimizer if not isinstance(opt, torch.optim.Optimizer)]
                raise TypeError(f"All elements in 'optimizer' list must be torch.optim.Optimizer, got {', '.join(bad_types)}")
        else:
            raise TypeError(
                f"'optimizer' must be a torch.optim.Optimizer or a list of 1-2 optimizers, "
                f"got {type(optimizer).__name__}"
            )
        if not isinstance(device, torch.device):
            raise TypeError(f"'device' must be a torch.device, got {type(device).__name__}")
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
            raise TypeError(f"'scheduler' must be a torch.optim.lr_scheduler.LRScheduler, got {type(scheduler).__name__}")
        
        self.scheduler = scheduler
        self.device = device

        self.episode_actions = []
        self.episode_rewards = []


    def save_action(self, action: torch.Tensor, prob: torch.Tensor, value: torch.Tensor):
        if not self.episode_actions:
            self.episode_actions = [[] for _ in range(action.shape[0])]
        
        if value.dim() == 1:
            value = value.unsqueeze(-1)

        for i in range(action.shape[0]):
            saved = SavedAction(
                action=action[i],
                prob=prob[i],
                value=value[i]
            )
            self.episode_actions[i].append(saved)


    def save_reward(self, reward):
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)
        
        if not self.episode_rewards:
            self.episode_rewards = [[] for _ in range(reward.shape[0])]

        for i in range(reward.shape[0]):
            self.episode_rewards[i].append(reward[i].item())


class A2C(_TrainingMethod):
    def __init__(self, optimizer, device, scheduler,
                gamma=0.9, 
                gae_lambda=1.0, 
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

        flattened_actions = [a for env_actions in self.episode_actions for a in env_actions]
        flattened_rewards = [r for env in self.episode_rewards for r in env]

        values = torch.stack([a.value for a in flattened_actions]).squeeze().to(device=self.device)
        actions = torch.stack([a.action for a in flattened_actions]).unsqueeze(1).to(device=self.device)
        probs = torch.stack([a.prob for a in flattened_actions]).squeeze(1).to(device=self.device)

        log_probs = torch.log(probs.gather(1, actions) + eps).squeeze(1) 
        rewards = torch.tensor(flattened_rewards, device=self.device, dtype=values.dtype)

        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        discounted_sum = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
        last_advantage = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
        for t in reversed(range(len(rewards))):
            discounted_sum = rewards[t] + self.gamma * discounted_sum
            returns[t] = discounted_sum
            if t < len(rewards) - 1:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            else:
                delta = rewards[t] - values[t] 
            advantages[t] = delta + self.gamma * self.gae_lambda * last_advantage
            last_advantage = advantages[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)
        advantages = torch.clamp(advantages, -10, 10)

        if torch.isnan(advantages).any() or torch.isinf(advantages).any():
            raise ValueError("NaN/Inf detected in advantages. Check rewards/values.")
        
        entropy = -(probs * probs.log()).sum(dim=1).mean()
        if torch.isnan(entropy) or torch.isinf(entropy):
            entropy = torch.tensor(0.0, device=entropy.device)
        self.entropy_coef = max(self.min_entropy_coef, self.entropy_coef * self.entropy_decay)
        entropy_bonus = self.entropy_coef * entropy

        actor_loss = (-log_probs * advantages).sum() - entropy_bonus
        critic_loss = F.mse_loss(values, returns)
        loss = actor_loss + critic_loss

        loss_time = time.perf_counter() - start_time
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            params = [p for group in self.optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(params, max_norm=self.max_grad_norm)
            self.optimizer.step()
        
        else:
            has_actor = hasattr(self, 'actor_optimizer') and self.actor_optimizer is not None
            has_critic = hasattr(self, 'critic_optimizer') and self.critic_optimizer is not None

            if not has_actor or not has_critic:
                raise RuntimeError("Both actor_optimizer and critic_optimizer must be set if single optimizer is not used.")
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            actor_params = [p for group in self.actor_optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(actor_params, max_norm=self.max_grad_norm)
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_params = [p for group in self.critic_optimizer.param_groups for p in group['params']]
            torch.nn.utils.clip_grad_norm_(critic_params, max_norm=self.max_grad_norm)
            self.critic_optimizer.step()


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
            'entropy_bonus': entropy_bonus.item(),
            'mean_advantage': advantages.mean().item(),
            'training_duration': duration,
            'loss_time': loss_time
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