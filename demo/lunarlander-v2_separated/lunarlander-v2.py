from jerl.TrainingAgent import TrainingAgent
import gymnasium as gym

import os
script_directory = os.path.dirname(os.path.abspath(__file__))

cfg = {
    "training_options": {
        "device": "cpu",
        "time_steps": 10000,
        "batch_size": 1,
        "output_dir": script_directory,
        "checkpoint_freq": -1,
        "end_condition": 5000,
        "visualize": True
    },
    "model":{
        "type": "separated_linear",
        "model_dims": [8, 64, 128, 256, 128, 64, 4],
        "critic_network_dims": [8, 64, 128, 256, 128, 64, 1], 
        "actor_activation_funct": "tanh", 
        "critic_activation_funct": "relu", 
        "dtype": "float",
        "use_layer_norm": True
    },
    "optimizer": {
        "actor": {
            "type": "Adam",
            "lr": 3e-2
        },
        "critic": {
            "type": "Adam",
            "lr": 1e-2
        }
    },
    "scheduler": None,
    "trainer": {
        "type": "A2C",
        "gamma": 0.99, 
        "gae_lambda": 0.95, 
        "initial_entropy_coef": 0.1, 
        "min_entropy_coef": 0.001, 
        "entropy_decay": 0.999, 
        "max_grad_norm": 0.5 
    }
}

env = gym.make("LunarLander-v2")

trainer = TrainingAgent(cfg=cfg)
trainer.train(env)