from jerl.ActorCriticNN import ActorCriticLNN
from jerl.TrainingMethods import A2C, SAC, PPO

import os
from pathlib import Path
from datetime import datetime
import time
from collections import defaultdict
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical


class TrainingAgent:


    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env 
        self.device = self._get_device()
        self.model = self._initialize_model()
        self.optimizer = self._get_optimizer()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=cfg.scheduler_step_size, 
            gamma=cfg.scheduler_gamma
        )
        self.trainer = self._get_trainer()
        self.metrics = defaultdict(list)


    def _get_device(self):
        if self.cfg.use_cuda:
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                raise RuntimeError("CUDA is set to True in the config, but CUDA is not available. Exiting.")
        else:
            return torch.device('cpu')


    def _initialize_model(self):
        if self.cfg.use_save_file:
            model = ActorCriticLNN(self.cfg.model_dims, device=self.device)
            model.load_state_dict(torch.load(self.cfg.model_save_file, map_location=self.device))
        else:
            model = ActorCriticLNN(self.cfg.model_dims, device=self.device)
        return model


    def _get_trainer(self):
        trainer_map = {'A2C': A2C, 'SAC': SAC, 'PPO': PPO}
        
        trainer_class = trainer_map.get(self.cfg.training_method)
        if trainer_class is None:
            raise ValueError(
                f"Unsupported training method: {self.cfg.training_method}. "
                f"Supported methods are: {list(trainer_map.keys())}"
            )
        
        return trainer_class(
            device=self.device,
            cfg=self.cfg,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )


    def _get_optimizer(self):
        optimizer_map = {
            'Adadelta': optim.Adadelta,
            'Adagrad': optim.Adagrad,
            'Adam': optim.Adam,
            'AdamW': optim.AdamW,
            'SparseAdam': optim.SparseAdam,
            'Adamax': optim.Adamax,
            'ASGD': optim.ASGD,
            'LBFGS': optim.LBFGS,
            'NAdam': optim.NAdam,
            'RAdam': optim.RAdam,
            'RMSprop': optim.RMSprop,
            'Rprop': optim.Rprop,
            'SGD': optim.SGD
        }

        if self.cfg.optimizer not in optimizer_map:
            raise ValueError(f"Unsupported optimizer: {self.cfg.optimizer}")

        optimizer_class = optimizer_map[self.cfg.optimizer]

        base_params = {
            'params': self.model.parameters(),
            'lr': self.cfg.learning_rate
        }

        param_mapping = {
            'Adadelta': ['rho', 'eps', 'weight_decay'],
            'Adagrad': ['lr_decay', 'weight_decay', 'initial_accumulator_value', 'eps'],
            'Adam': ['betas', 'eps', 'weight_decay', 'amsgrad'],
            'AdamW': ['betas', 'eps', 'weight_decay', 'amsgrad'],
            'SparseAdam': ['betas', 'eps'],
            'Adamax': ['betas', 'eps', 'weight_decay'],
            'ASGD': ['lambd', 'alpha', 't0', 'weight_decay'],
            'NAdam': ['betas', 'eps', 'weight_decay', 'momentum_decay'],
            'RAdam': ['betas', 'eps', 'weight_decay'],
            'RMSprop': ['alpha', 'eps', 'weight_decay', 'momentum', 'centered'],
            'Rprop': ['etas', 'step_sizes'],
            'SGD': ['momentum', 'dampening', 'weight_decay', 'nesterov']
        }

        params = base_params.copy()
        for param_name in param_mapping[self.cfg.optimizer]:
            if not hasattr(self.cfg, param_name):
                raise ValueError(f"Missing required parameter {param_name} for optimizer {self.cfg.optimizer}")
            params[param_name] = getattr(self.cfg, param_name)

        return optimizer_class(**params)


    def _run_episode(self):
        print("Running Episode...")
        start_time = time.perf_counter()

        episode_reward = 0
        
        for i in range(self.cfg.time_steps):
            if i % (self.cfg.time_steps) == 0:
                state, _ = self.env.reset()
            action = self._select_action(state)
            state, reward, done, _, _ = self.env.step(action)
            self.actor_critic.episode_rewards.append(reward)
            episode_reward += reward
            if done:
                break
                
        duration = time.perf_counter() - start_time
        print(f"Episode Complete in {duration:.2f}s.")

        return episode_reward
    

    def _select_action(self, state):
        state = np.ndarray.flatten(state)
        action_probabilities, state_value = self.model(state)
        m = Categorical(action_probabilities)
        action = m.sample()
        self.actor_critic.save_action(action, action_probabilities, state_value)
        return action.item()
    

    def train(self):
        print("Beginning Training...")

        episode_num = 0
        
        while True:
            episode_num += 1
            episode_reward = self._run_episode()
            
            # Store metrics
            self.metrics['rewards'].append(episode_reward)
            
            # Training step
            self.actor_critic.train(self.optimizer, self.scheduler, self.cfg)
            self._log_progress(episode_num)
            
            # Checkpointing
            if episode_num % self.cfg.checkpoint_freq == 0:
                self._save_checkpoint(episode_num)
                
            # Termination condition
            if episode_reward > self.cfg.end_condition:
                self._finalize_training(episode_num)
                break
    

    def _log_progress(self, episode_num):
        reward = self.metrics['rewards'][-1]
        avg_reward = np.mean(self.metrics['rewards'][-10:])
        print(f'Episode {episode_num} | '
              f'Reward: {reward:.2f} | '
              f'Average Reward: {avg_reward:.2f}')


    def _save_checkpoint(self, episode_num):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = f"{script_dir}/saved_models/episode_{episode_num}.pth"
        torch.save(self.model.state_dict(), filepath)
    

    def _finalize_training(self, episode_num):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_reward = self.metrics['rewards'][-1]    
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_folder = Path(f"{script_dir}/saved_models/final")
        filename = (
            f"final_model_"
            f"ep{episode_num}_"
            f"rew{final_reward:.0f}_"
            f"target{self.cfg.end_condition}_"
            f"{timestamp}.pth"
        )
        full_path = save_folder / filename
        
        torch.save(self.model.state_dict(), full_path)
        print(f"Saved final model as: {full_path}")