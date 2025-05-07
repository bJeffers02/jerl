class Config(dict):
    required_fields = {
        'checkpoint_freq': int,
        'end_condition': int,
        'reward_gamma': float,
        'scheduler_gamma': float,
        'scheduler_step_size': int,
        'learning_rate': float,
        'initial_entropy_coef': float,
        'min_entropy_coef': float,
        'entropy_decay': float,
        'use_save_file': bool,
        'model_save_file': str,
        'model_dims': list,
        'use_cuda': bool,
        'time_steps': int,
        'optimizer': str,
        'training_method': str,
        
        # Adam/AdamW parameters
        'adam_beta1': float,
        'adam_beta2': float,
        'adam_eps': float,
        'weight_decay': float,
        'adam_amsgrad': bool,
        
        # SGD parameters
        'sgd_momentum': float,
        'sgd_dampening': float,
        'sgd_nesterov': bool,
        
        # RMSprop parameters
        'rmsprop_alpha': float,
        'rmsprop_momentum': float,
        'rmsprop_centered': bool,
        
        # Adadelta parameters
        'adadelta_rho': float,
        'adadelta_eps': float,
        
        # Adagrad parameters
        'adagrad_lr_decay': float,
        'adagrad_initial_accumulator_value': float,
        'adagrad_eps': float,
        
        # ASGD parameters
        'asgd_lambd': float,
        'asgd_alpha': float,
        'asgd_t0': float,
        
        # LBFGS parameters
        'lbfgs_max_iter': int,
        'lbfgs_max_eval': int,
        'lbfgs_tolerance_grad': float,
        'lbfgs_tolerance_change': float,
        'lbfgs_history_size': int,
        
        # NAdam parameters
        'nadam_momentum_decay': float,
        
        # Rprop parameters
        'rprop_eta1': float,
        'rprop_eta2': float,
        'rprop_step_size1': float,
        'rprop_step_size2': float
    }

    def __init__(self, config_dict: dict):
        self._validate_and_set(config_dict)

    def _validate_and_set(self, config_dict: dict):
        missing_keys = [key for key in self.required_fields if key not in config_dict]
        extra_keys = [key for key in config_dict if key not in self.required_fields]

        if missing_keys:
            raise ValueError(f"Missing required config fields: {missing_keys}")
        if extra_keys:
            raise ValueError(f"Unexpected config fields: {extra_keys}")

        for key, expected_type in self.required_fields.items():
            value = config_dict[key]
            if not isinstance(value, expected_type):
                raise TypeError(f"Field '{key}' must be of type {expected_type.__name__}, got {type(value).__name__}")
            self[key] = value

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __setitem__(self, key, value):
        if key in self.required_fields:
            expected_type = self.required_fields[key]
            if not isinstance(value, expected_type):
                raise TypeError(f"Field '{key}' must be of type {expected_type.__name__}, got {type(value).__name__}")
            super().__setitem__(key, value)
        else:
            raise KeyError(f"Unexpected config field: '{key}'")

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __repr__(self):
        return f"Config({dict(self)})"