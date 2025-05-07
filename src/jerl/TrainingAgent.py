from jerl.ActorCriticNN import ActorCriticLNN
from jerl.TrainingMethods import A2C, SAC, PPO

import os
from pathlib import Path
from datetime import datetime
import time
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical


support_map = {
    'model_map' : {
        'ACLNN' : ActorCriticLNN
    }, 
    'optimizer_map' : {
        'Adam': optim.Adam,
        'AdamW': optim.AdamW,
        'RMSprop': optim.RMSprop,
        'SGD': optim.SGD,
        'NAdam': optim.NAdam,
        'RAdam': optim.RAdam
    },
    'scheduler_map' : {
        'StepLR' : optim.lr_scheduler.StepLR
    },
    'trainer_map' : {
        'A2C': A2C,
        'SAC': SAC, 
        'PPO': PPO
    }
}


class TrainingAgent:
    

    def __init__(self, cfg):
        
        
        self.cfg = cfg
        self.device = _get_device()
        self.model = _get_model()
        self.optimizer = _get_optimizer()
        self.scheduler = _get_scheduler()
        self.trainer = _get_trainer()


        def _get_device():
            if self.cfg.use_cuda:
                if torch.cuda.is_available():
                    return torch.device('cuda')
                else:
                    raise RuntimeError("CUDA is set to True in the config, but CUDA is not available. Exiting.")
            else:
                return torch.device('cpu')


        def _get_model():
            model_map = support_map.model_map

            model_options = self.cfg.get('model', {})
            type = model_options.get('type', 'ACLNN')
            model_dims = model_options.get('model_dims', [])

            if type not in model_map:
                raise ValueError(
                    f"Unsupported model type: {type}. "
                    f"Supported model types are: {list(model_map.keys())}"
                )

            model_cls = model_map[type]

            kwargs = {k: v for k, v in model_options.items() if k not in ('type', 'model_dims')}
            
            return model_cls(model_dims, self.device, **kwargs)


        def _get_optimizer():
            optimizer_map = support_map.optimizer_map

            optimizer_options = self.cfg.get('optimizer', {})
            type = optimizer_options.get('type', 'Adam')
            lr = optimizer_options.get('lr', 3e-4)

            if type not in optimizer_map:
                raise ValueError(
                    f"Unsupported optimizer: {type}. "
                    f"Supported optimizers are: {list(optimizer_map.keys())}"
                )

            optimizer_cls = optimizer_map[type]

            kwargs = {k: v for k, v in optimizer_options.items() if k not in ('type', 'lr')}

            return optimizer_cls(self.model.parameters, lr=lr, **kwargs)


        def _get_scheduler():
            scheduler_map = support_map.scheduler_map

            scheduler_options = self.cfg.get('scheduler', {})
            type = scheduler_options.get('type', 'StepLR')
            step_size = scheduler_options.get('step_size', 100)
            gamma = scheduler_options.get('gamma', 0.9)

            if type not in scheduler_map:
                raise ValueError(
                    f"Unsupported scheduler: {type}. "
                    f"Supported scheduler are: {list(scheduler_map.keys())}"
                )

            scheduler_cls = scheduler_map[type]

            kwargs = {k: v for k, v in scheduler_options.items() if k not in ('type', 'step_size', 'gamma')}

            return scheduler_cls(self.optimizer, step_size=step_size, gamma=gamma, **kwargs)

        
        def _get_trainer():
            trainer_map = support_map.trainer_map
            
            trainer_options = self.cfg.get('trainer', {})
            type = trainer_options.get('type', 'A2C')
            gamma = trainer_options.get('gamma', 0.9)

            if type not in trainer_map:
                raise ValueError(
                    f"Unsupported training method: {type}. "
                    f"Supported methods are: {list(trainer_map.keys())}"
                )
            
            trainer_cls = trainer_map[type]

            kwargs = {k: v for k, v in trainer_options.items() if k not in ('type', 'gamma')}

            return trainer_cls(self.device, self.optimizer, self.scheduler, gamma, **kwargs)


    def _run_episode(self, env):
        print("Running Episode...")
        start_time = time.perf_counter()

        episode_reward = 0
        
        for i in range(self.cfg.time_steps):
            if i % (self.cfg.time_steps) == 0:
                state, _ = env.reset()
            action = self._select_action(state)
            state, reward, done, _, _ = env.step(action)
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
    

    def train(self, env):
        print("Beginning Training...")

        episode_num = 0
        
        while True:
            episode_num += 1
            episode_reward = self._run_episode(env)
            
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