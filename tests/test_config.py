import unittest
from jerl.Configuration import Config

VALID_CONFIG = {
    'checkpoint_freq': 100,
    'end_condition': 1000,
    'reward_gamma': 0.99,
    'scheduler_gamma': 0.8,
    'scheduler_step_size': 100,
    'learning_rate': 0.005,
    'initial_entropy_coef': 0.1,
    'min_entropy_coef': 0.001,
    'entropy_decay': 1.0,
    'use_save_file': False,
    'model_save_file': "saved_models/episode_600.pth",
    'model_dims': [120, 512, 1024, 512, 128, 5],
    'use_cuda': False,
    'time_steps': 288
}


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(VALID_CONFIG)

    def test_attribute_access(self):
        self.assertEqual(self.cfg.checkpoint_freq, 100)
        self.assertEqual(self.cfg['checkpoint_freq'], 100)

    def test_attribute_assignment(self):
        self.cfg.learning_rate = 0.001
        self.assertEqual(self.cfg.learning_rate, 0.001)
        self.cfg['entropy_decay'] = 0.8
        self.assertEqual(self.cfg.entropy_decay, 0.8)

    def test_type_check_on_assignment(self):
        with self.assertRaises(TypeError):
            self.cfg.reward_gamma = "not a float"

        with self.assertRaises(TypeError):
            self.cfg['use_save_file'] = "not a bool"

    def test_unknown_key_assignment(self):
        with self.assertRaises(KeyError):
            self.cfg['unknown'] = 123

        with self.assertRaises(KeyError):
            self.cfg.unknown = 123

    def test_repr(self):
        self.assertIn("Config(", repr(self.cfg))
        self.assertIn("'checkpoint_freq': 100", repr(self.cfg))

    def test_missing_keys(self):
        partial = VALID_CONFIG.copy()
        del partial['reward_gamma']
        with self.assertRaises(ValueError):
            Config(partial)

    def test_extra_keys(self):
        extra = VALID_CONFIG.copy()
        extra['bad_field'] = 123
        with self.assertRaises(ValueError):
            Config(extra)

    def test_wrong_type_initialization(self):
        bad = VALID_CONFIG.copy()
        bad['model_dims'] = "not a list"
        with self.assertRaises(TypeError):
            Config(bad)


if __name__ == '__main__':
    unittest.main()
