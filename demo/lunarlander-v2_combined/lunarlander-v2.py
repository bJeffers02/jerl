from jerl.TrainingAgent import TrainingAgent
import gymnasium as gym

import os
script_directory = os.path.dirname(os.path.abspath(__file__))

cfg = {
    "training_options": {
        "device": "cpu",
        "time_steps": 1000,
        "env_vector_size": 10,
        "output_dir": script_directory,
        "checkpoint_freq": -1,
        "end_condition": 5000,
        "visualize": True
    },
    "model":{
        "type": "combined_linear",
        "model_dims": [8, 64, 128, 256, 512, 256, 128, 64, 4],
        "activation_funct": "tanh",
        "dtype": "float",
        "use_layer_norm": True
    },
    "optimizer": {
        "type": "Adam",
        "lr": 3e-4,
        "eps": 1e-5
    },
    "scheduler": None,
    "trainer": {
        "type": "A2C",
        "gamma": 0.95, 
        "gae_lambda": 0.95, 
        "initial_entropy_coef": 0.5, 
        "min_entropy_coef": 0.001, 
        "entropy_decay": 0.995, 
        "max_grad_norm": 0.7 
    }
}

def make_env():
    return gym.make("LunarLander-v2")

trainer = TrainingAgent(cfg=cfg)
trainer.train(make_env)