from ActorCriticNN import ActorCriticNN

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

    def __init__(self, cfg, reward_funct, create_env, device):
        print("Creating Training Agent...")
        self.device = device
        self.cfg = cfg
        self.env = create_env(reward_funct, self.cfg)
        self.actor_critic = self._initialize_model()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=cfg.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=cfg.scheduler_step_size, 
            gamma=cfg.scheduler_gamma
        )
        self.metrics = defaultdict(list)

        print("Training Agent Created.")

    def _initialize_model(self):
        if self.cfg.use_save_file:
            model = ActorCriticNN(self.cfg.model_dims, device=self.device)
            model.load_state_dict(torch.load(self.cfg.model_save_file, map_location=self.device))
        else:
            model = ActorCriticNN(self.cfg.model_dims, device=self.device)
        return model
    
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
        action_probabilities, state_value = self.actor_critic(state)
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
        torch.save(self.actor_critic.state_dict(), filepath)
    
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
        
        torch.save(self.actor_critic.state_dict(), full_path)
        print(f"Saved final model as: {full_path}")