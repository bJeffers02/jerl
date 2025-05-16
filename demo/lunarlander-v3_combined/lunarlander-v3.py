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
        "type": "combined_linear",
        "model_dims": [8, 256, 256, 128, 4],
        "activation_funct": "tanh",
        "dtype": "float",
        "use_layer_norm": True
    },
    "optimizer": {
        "type": "Adam",
        "lr": 1e-4,
        "eps": 1e-5
    },
    "scheduler": None,
    "trainer": {
        "type": "A2C",
        "gamma": 0.93, 
        "gae_lambda": 0.95, 
        "initial_entropy_coef": 0.1, 
        "min_entropy_coef": 0.001, 
        "entropy_decay": 0.995, 
        "max_grad_norm": 0.7 
    }
}

env = gym.make("LunarLander-v2", continuous=False, gravity=-10.0,
               enable_wind=True, wind_power=5.0, turbulence_power=0.5)

trainer = TrainingAgent(cfg=cfg)
trainer.train(env)