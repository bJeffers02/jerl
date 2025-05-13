from jerl.TrainingAgent import TrainingAgent
import gymnasium as gym

cfg = {
    "training_options": {
        "device": "cpu",
        "time_steps": 500,
        "batch_size": 10,
        "output_dir": "/home/benjaminj/GIT/jerl/demo",
        "checkpoint_freq": 10000,
        "end_condition": 500
    },
    "model":{
        "type": "combined_linear",
        "model_dims": [4, 64, 128, 64, 2]
    },
    "optimizer": {
        "type": "Adam",
        "lr": 3e-3
    },
    "scheduler": {
        "type": "StepLR",
        "step_size": 100,
        "gamma": 1.0
    },
    "trainer": {
        "type": "A2C",
        "entropy_decay": 0.9999
    }
}

trainer = TrainingAgent(cfg=cfg)
env = gym.make("CartPole-v1")

trainer.train(env)