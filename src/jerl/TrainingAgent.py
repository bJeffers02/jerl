from jerl.ActorCriticNN import CombinedActorCriticLinear, SeparatedActorCriticLinear
from jerl.TrainingMethods import A2C

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is not installed. Please install it manually:\n"
        "CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "Or visit https://pytorch.org/get-started/locally for CUDA options."
    )

import csv
from pathlib import Path
from datetime import datetime
import time
import numpy as np
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
            model_map = support_map.get('model_map')

            model_options = self.cfg.get('model', {})
            model_type = model_options.get('type', 'combined_linear')
            model_dims = model_options.get('model_dims', [])

            if model_type not in model_map:
                raise ValueError(
                    f"Unsupported model type: {model_type}. "
                    f"Supported model types are: {list(model_map.keys())}"
                )

            model_cls = model_map[model_type]

            kwargs = {k: v for k, v in model_options.items() if k not in ('type', 'model_dims')}
            
            return model_cls(model_dims, device=self.device, **kwargs)


        def _get_optimizer():
            optimizer_map = support_map.get('optimizer_map')

            optimizer_options = self.cfg.get('optimizer', {})
            optimizer_type = optimizer_options.get('type', 'Adam')
            lr = optimizer_options.get('lr', 3e-4)

            if optimizer_type not in optimizer_map:
                raise ValueError(
                    f"Unsupported optimizer: {optimizer_type}. "
                    f"Supported optimizers are: {list(optimizer_map.keys())}"
                )

            optimizer_cls = optimizer_map[optimizer_type]

            kwargs = {k: v for k, v in optimizer_options.items() if k not in ('type', 'lr')}

            return optimizer_cls(self.model.parameters(), lr=lr, **kwargs)


        def _get_scheduler():
            scheduler_map = support_map.get('scheduler_map')

            scheduler_options = self.cfg.get('scheduler', {})
            scheduler_type = scheduler_options.get('type', 'StepLR')
            step_size = scheduler_options.get('step_size', 100)
            gamma = scheduler_options.get('gamma', 0.9)

            if scheduler_type not in scheduler_map:
                raise ValueError(
                    f"Unsupported scheduler: {scheduler_type}. "
                    f"Supported scheduler are: {list(scheduler_map.keys())}"
                )

            scheduler_cls = scheduler_map[scheduler_type]

            kwargs = {k: v for k, v in scheduler_options.items() if k not in ('type', 'step_size', 'gamma')}

            return scheduler_cls(self.optimizer, step_size=step_size, gamma=gamma, **kwargs)

        
        def _get_trainer():
            trainer_map = support_map.get('trainer_map')
            
            trainer_options = self.cfg.get('trainer', {})
            trainer_type = trainer_options.get('type', 'A2C')

            if trainer_type not in trainer_map:
                raise ValueError(
                    f"Unsupported training method: {trainer_type}. "
                    f"Supported methods are: {list(trainer_map.keys())}"
                )
            
            trainer_cls = trainer_map[trainer_type]

            kwargs = {k: v for k, v in trainer_options.items() if k not in ('type')}

            return trainer_cls(self.optimizer, self.device, self.scheduler, **kwargs)
        

        self.cfg = cfg
        self.training_options = self.cfg.get('training_options')
        self.device = _get_device()
        self.model = _get_model()
        self.optimizer = _get_optimizer()
        self.scheduler = _get_scheduler()
        self.trainer = _get_trainer()

        self.full_metrics = []

    
    def train(self, env):


        def _run_episode(env):


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
            for i in range(self.training_options.get('time_steps') * self.training_options.get('batch_size')):
                if i % (self.training_options.get('time_steps')) == 0:
                    state, _ = env.reset()
                action = _select_action(state)
                state, reward, done, _, _ = env.step(action)
                self.trainer.save_reward(reward)
                episode_reward += reward
                if done:
                    state, _ = env.reset()
                    
            duration = time.perf_counter() - start_time
            print(f"Episode Complete in {duration:.2f}s.")

            metrics = {
                "episode_reward": episode_reward,
                "episode_duration": duration
            }
            return metrics


        def _log_progress(episode_num, metrics):
            metrics_rounded = {k: round(v, 2) for k, v in metrics.items()}
            output = ' | '.join(f"{k}: {v}" for k, v in metrics_rounded.items())
            print(f'Episode {episode_num} | {output}')


        def _save_checkpoint(episode_num):
            output_dir = Path(self.training_options.get('output_dir'))
            checkpoint_dir = output_dir / 'saved_models'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            filepath = checkpoint_dir / f"episode_{episode_num}.pth"
            torch.save(self.model.state_dict(), filepath)


        def _finalize_training(final_reward):
            print("Finished Training.")

            env_name = env.spec.id
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 
            save_folder = Path(self.training_options.get('output_dir')) / 'saved_models' / 'final'
            save_folder.mkdir(parents=True, exist_ok=True)
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
            episode_metrics = _run_episode(env)
            
            training_metrics = self.trainer.train()
            combined_metrics = episode_metrics | training_metrics
        
            csv_file = Path(f"{self.training_options.get('output_dir')}/metrics.csv")
            with open(csv_file, 'a', newline='') as csvfile:
                fieldnames = list(combined_metrics.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if csvfile.tell() == 0:
                    writer.writeheader()
                writer.writerow(combined_metrics)

            _log_progress(episode_num, combined_metrics)
            

            # Checkpointing
            if episode_num % self.training_options.get('checkpoint_freq') == 0:
                _save_checkpoint(episode_num)
                
            # Termination condition
            if combined_metrics.get('episode_reward') >= self.training_options.get('end_condition'):
                _finalize_training(combined_metrics.get('episode_reward'))
                break   