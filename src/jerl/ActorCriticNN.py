import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


def _validate_network_dims(name, dims):
        if not isinstance(dims, list):
            raise TypeError(f"'{name}' must be a list, but got {type(dims).__name__}.")
        if len(dims) < 2:
            raise ValueError(f"'{name}' must have at least two dimensions, but got {len(dims)}.")
        if not all(isinstance(dim, int) and dim > 0 for dim in dims):
            raise ValueError(f"All elements of '{name}' must be positive integers. Got: {dims}")


def _validate_activation_funct(name, activation_funct):
    activation_map = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'elu': nn.ELU(),
        'gelu': nn.GELU(),
    }      
    if activation_funct is None or activation_funct == 'none':
        return nn.Identity()
    elif isinstance(activation_funct, str):
        if activation_funct not in activation_map:
            raise ValueError(
                f"Unsupported activation function for '{name}': {activation_funct}. "
                f"Supported options are: {list(activation_map.keys())}"
            )
        return activation_map[activation_funct]
    elif isinstance(activation_funct, nn.Module):
        return activation_funct
    else:
        raise TypeError(
            f"'{name}' must be a string (e.g. {list(activation_map.keys())}) "
            f"or an instance of nn.Module, but got {type(activation_funct).__name__}."
        )

  
def _validate_dtype(dtype):
    allowed_dtypes = {
        'float16': torch.float16,
        'half': torch.half,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
        'float': torch.float,
        'float64': torch.float64,
        'double': torch.double,
        torch.float16: torch.float16,
        torch.half: torch.half,
        torch.bfloat16: torch.bfloat16,
        torch.float32: torch.float32,
        torch.float: torch.float,
        torch.float64: torch.float64,
        torch.double: torch.double,
    }

    if isinstance(dtype, str):
        key = dtype.lower().strip()
        if key not in allowed_dtypes:
            raise ValueError(f"Unsupported dtype string: '{dtype}'. Allowed: {list(k for k in allowed_dtypes if isinstance(k, str))}")
        return allowed_dtypes[key]
    elif isinstance(dtype, torch.dtype):
        if dtype not in allowed_dtypes:
            raise ValueError(f"Unsupported torch.dtype: {dtype}. Allowed: {list(k for k in allowed_dtypes if isinstance(k, torch.dtype))}")
        return dtype
    else:
        raise TypeError(f"'dtype' must be a string or torch.dtype, but got {type(dtype).__name__}.")


class CombinedActorCriticLinear(nn.Module):
    def __init__(self, network_dims, activation_funct='relu', device=None, dtype=torch.float32, **kwargs):
        super(CombinedActorCriticLinear, self).__init__()

        if kwargs:
            for key in kwargs:
                print(f"Unexpected argument: '{key}' with value '{kwargs[key]}'")

        _validate_network_dims("actor_network_dims", network_dims)

        self.activation_funct = _validate_activation_funct("activation_funct", activation_funct)
        
        layers = []
        for i in range(len(network_dims) - 2):
            layers.append(nn.Linear(network_dims[i], network_dims[i+1]))
            layers.append(copy.deepcopy(self.activation_funct))
        
        self.hidden_layers = nn.Sequential(*layers)

        self.actor_layer = nn.Linear(network_dims[-2], network_dims[-1])
        self.critic_layer = nn.Linear(network_dims[-2], 1)
        
        if device is not None and not isinstance(device, torch.device):
            raise TypeError(f"'device' must be a torch.device, got {type(device).__name__}")
        self.device = device if device is not None else torch.device('cpu')

        self.dtype = _validate_dtype(dtype)

        self.to(device=self.device, dtype=self.dtype)


    def forward(self, x):
        if not isinstance(x, (np.ndarray, torch.Tensor)):
            raise TypeError(f"Input must be a numpy array or torch tensor, got {type(x).__name__}.")
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.to(device=self.device, dtype=self.dtype)
        
        x = self.hidden_layers(x)
        
        action = F.softmax(self.actor_layer(x), dim=-1)
        state_value = self.critic_layer(x)
        
        return action, state_value
    
class SeparatedActorCriticLinear(nn.Module):
    def __init__(self, actor_network_dims, critic_network_dims=None, actor_activation_funct='relu', critic_activation_funct='relu', device=None, dtype=torch.float32, **kwargs):
        super(SeparatedActorCriticLinear, self).__init__()

        if kwargs:
            for key in kwargs:
                print(f"Unexpected argument: '{key}' with value '{kwargs[key]}'")

        if critic_network_dims is None:
            critic_network_dims = actor_network_dims
            critic_network_dims[-1] = 1

        _validate_network_dims("actor_network_dims", actor_network_dims)
        _validate_network_dims("critic_network_dims", critic_network_dims)

        if actor_network_dims[0] != critic_network_dims[0]:
            raise ValueError(
                f"Input dimension mismatch: actor_network_dims[0] = {actor_network_dims[0]} "
                f"but critic_network_dims[0] = {critic_network_dims[0]}. "
                "Both must match the observation space dimension."
            )

        self.actor_activation_funct = _validate_activation_funct("actor_activation_funct", actor_activation_funct)
        self.critic_activation_funct = _validate_activation_funct("critic_activation_funct", critic_activation_funct)

        actor_layers = []
        for i in range(len(actor_network_dims) - 1):
            actor_layers.append(nn.Linear(actor_network_dims[i], actor_network_dims[i+1]))
            actor_layers.append(copy.deepcopy(self.actor_activation_funct))
        
        self.actor_network = nn.Sequential(*actor_layers)

        critic_layers = []
        for i in range(len(critic_network_dims) - 1):
            critic_layers.append(nn.Linear(critic_network_dims[i], critic_network_dims[i+1]))
            critic_layers.append(copy.deepcopy(self.critic_activation_funct))
        if critic_network_dims[-1] != 1:
            critic_layers.append(nn.Linear(critic_network_dims[-1], 1))

        self.critic_network = nn.Sequential(*critic_layers)
        
        if device is not None and not isinstance(device, torch.device):
            raise TypeError(f"'device' must be a torch.device, got {type(device).__name__}")
        self.device = device if device is not None else torch.device('cpu')

        self.dtype = _validate_dtype(dtype)

        self.to(device=self.device, dtype=self.dtype)


    def forward(self, x):
        if not isinstance(x, (np.ndarray, torch.Tensor)):
            raise TypeError(f"Input must be a numpy array or torch tensor, got {type(x).__name__}.")
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.to(device=self.device, dtype=self.dtype)
        
        action = F.softmax(self.actor_network(x), dim=-1)
        state_value = self.critic_network(x)
        
        return action, state_value