from jerl.ActorCriticNN import CombinedActorCriticLinear, SeparatedActorCriticLinear
from jerl.TrainingMethods import A2C

from pathlib import Path
from datetime import datetime
import time
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical


support_map = {
    'model_map' : {
        'combined_linear' : CombinedActorCriticLinear,
        'separated_linear' : SeparatedActorCriticLinear
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
        'A2C': A2C
    }
}


class TrainingAgent:
    

    def __init__(self, cfg):
        
        
        self.cfg = cfg
        self.training_options = self.cfg.training_options
        self.device = _get_device()
        self.model = _get_model()
        self.optimizer = _get_optimizer()
        self.scheduler = _get_scheduler()
        self.trainer = _get_trainer()


        def _get_device():
            device = self.training_options.get('device', 'cpu')
           
            if not isinstance(device, str):
                raise TypeError(f"'device' must be a string, but got {type(device).__name__}.")

            device = device.lower().strip()

            if device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise RuntimeError(f"'device' is set to '{device}', but CUDA is not available.")
                if ':' in device:
                    try:
                        index = int(device.split(':')[1])
                        if index >= torch.cuda.device_count():
                            raise ValueError(f"Requested CUDA device index {index}, but only {torch.cuda.device_count()} available.")
                    except ValueError:
                        raise ValueError(f"Invalid CUDA device format: '{device}'. Expected 'cuda' or 'cuda:<index>'.")
                return torch.device(device) 
            elif device == "cpu":
                return torch.device('cpu')
            else:
                raise ValueError(f"Unsupported device: {device}. Supported: 'cpu', 'cuda', 'cuda:<index>'") 


        def _get_model():
            model_map = support_map.model_map

            model_options = self.cfg.get('model', {})
            type = model_options.get('type', 'combined_linear')
            model_dims = model_options.get('model_dims', [])

            if type not in model_map:
                raise ValueError(
                    f"Unsupported model type: {type}. "
                    f"Supported model types are: {list(model_map.keys())}"
                )

            model_cls = model_map[type]

            kwargs = {k: v for k, v in model_options.items() if k not in ('type', 'model_dims')}
            
            return model_cls(model_dims, device=self.device, **kwargs)


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

            return optimizer_cls(self.model.parameters(), lr=lr, **kwargs)


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

            if type not in trainer_map:
                raise ValueError(
                    f"Unsupported training method: {type}. "
                    f"Supported methods are: {list(trainer_map.keys())}"
                )
            
            trainer_cls = trainer_map[type]

            kwargs = {k: v for k, v in trainer_options.items() if k not in ('type')}

            return trainer_cls(self.device, self.optimizer, self.scheduler, **kwargs)

    
    def train(self, env):


        def _run_episode():


            def _select_action(state):
                state = np.ndarray.flatten(state)
                action_probabilities, state_value = self.model(state)
                m = Categorical(action_probabilities)
                action = m.sample()
                self.trainer.save_action(action, action_probabilities, state_value)
                return action.item()
            

            print("Running Episode...")
            start_time = time.perf_counter()

            episode_reward = 0
            for i in range(self.cfg.time_steps):
                if i % (self.cfg.time_steps) == 0:
                    state, _ = env.reset()
                action = _select_action(state)
                state, reward, done, _, _ = env.step(action)
                self.trainer.save_reward(reward)
                episode_reward += reward
                if done:
                    break
                    
            duration = time.perf_counter() - start_time
            print(f"Episode Complete in {duration:.2f}s.")

            return episode_reward


        def _log_progress(episode_num):
            reward = self.metrics['rewards'][-1]
            print(f'Episode {episode_num} | '
                  f'Reward: {reward:.2f}')


        def _save_checkpoint(episode_num):
            filepath = f"{self.cfg.output_dir}/saved_models/episode_{episode_num}.pth"
            torch.save(self.model.state_dict(), filepath)


        def _finalize_training(final_reward):
            print("Finished Training.")

            env_name = env.spec.id
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
            save_folder = Path(f"{self.cfg.output_dir}/saved_models/final")
            filename = (
                f"final_model_"
                f"{env_name}_"
                f"rew{final_reward:.0f}_"
                f"{timestamp}.pth"
            )
            full_path = save_folder / filename
            
            torch.save(self.model.state_dict(), full_path)
            print(f"Saved final model as: {full_path}")


        print("Beginning Training...")

        episode_num = 0 
        while True:
            episode_num += 1
            episode_reward = _run_episode(env)
            
            # Training step
            self.trainer.train()
            _log_progress(episode_num)
            
            # Checkpointing
            if episode_num % self.cfg.checkpoint_freq == 0:
                _save_checkpoint(episode_num)
                
            # Termination condition
            if episode_reward > self.cfg.end_condition:
                _finalize_training(episode_num)
                break   