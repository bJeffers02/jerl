from jerl.TrainingAgent import TrainingAgent
import gymnasium as gym
from gymnasium import RewardWrapper

import os
script_directory = os.path.dirname(os.path.abspath(__file__))

cfg = {
    "training_options": {
        "device": "cpu",
        "time_steps": 100,
        "batch_size": 5,
        "output_dir": script_directory,
        "checkpoint_freq": -1,
        "end_condition": 480,
        "visualize": True
    },
    "model":{
        "type": "combined_linear",
        "model_dims": [4, 64, 64, 2],
        "activation_funct": "relu",
        "dtype": "float",
        "use_layer_norm": True
    },
    "optimizer": {
        "type": "Adam",
        "lr": 3e-4
    },
    "scheduler": {
        "type": "StepLR",
        "step_size": 100,
        "gamma": 1.0
    },
    "trainer": {
        "type": "A2C",
        "gamma": 0.9, 
        "gae_lambda": 1.0, 
        "initial_entropy_coef": 0.1, 
        "min_entropy_coef": 0.001, 
        "entropy_decay": 0.99, 
        "max_grad_norm": 1.0 
    }
}

"""
Custom CartPole-v1 reward wrapper for Gymnasium.

The default CartPole environment gives a constant +1 reward per step, making the total
episode reward equal to the number of steps (which provides no meaningful gradient signal
for policy improvement). This wrapper modifies the reward function to give 
angle-proportional rewards: 
   - Reward = 1.0 - abs(theta)/0.2095
   - Maximum reward (1.0) when pole is perfectly vertical (theta=0)
   - Minimum reward (0.0) when pole reaches threshold angle (|theta|=0.2095 radians)

This creates a denser, more meaningful reward signal that:
- Encourages keeping the pole centered (not just surviving)
- Provides immediate feedback about pole stability
- Maintains compatibility with standard Actor-Critic algorithms
"""
class cartpole(RewardWrapper):
    def __init__(self):
        env = gym.make("CartPole-v1")
        super().__init__(env)
    def step(self, action):
        obs, reward, terminated, truncated, info  = self.env.step(action)
        
        _, _, theta, _ = obs
        reward = 1.0 - abs(theta) / 0.2095 + 0.01  
        
        if terminated:
            reward = -1.0
            
        return obs, reward, terminated, truncated, info 

env = cartpole()

trainer = TrainingAgent(cfg=cfg)
trainer.train(env)