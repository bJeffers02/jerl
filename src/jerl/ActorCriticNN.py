import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticLNN(nn.Module):
    def __init__(self, network_dims, activation_funct='relu', device=None):
        super(ActorCriticLNN, self).__init__()

        if not isinstance(network_dims, list):
            raise TypeError(f"'network_dims' must be a list, but got {type(network_dims).__name__}.")
        if len(network_dims) < 2:
            raise ValueError(f"'network_dims' must have at least two dimensions, but got {len(network_dims)}.")
        if not all(isinstance(dim, int) and dim > 0 for dim in network_dims):
            raise ValueError(f"All elements of 'network_dims' must be positive integers. Got: {network_dims}")

        activation_map = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
        }

        if activation_funct is None or activation_funct == 'none':
            self.activation_funct = nn.Identity()
        elif isinstance(activation_funct, str):
            if activation_funct not in activation_map:
                    raise ValueError(
                        f"Unsupported activation function: {activation_funct}. "
                        f"Supported activation function are: {list(activation_map.keys())}"
                    )
            self.activation_funct = activation_map[activation_funct]
        elif isinstance(activation_funct, nn.Module):
            self.activation_funct = activation_funct
        else:
            raise TypeError(
                f"'activation_funct' must be either a string (e.g. {list(activation_map.keys())}) "
                f"or an instance of nn.Module, but got {type(activation_funct).__name__}."
            )
        
        self.layers = nn.ModuleList()
        for i in range(len(network_dims) - 2):
            self.layers.append(nn.Linear(network_dims[i], network_dims[i+1]))
            self.layers.append(F.leaky_relu)
        
        self.actor_layer = nn.Linear(network_dims[-2], network_dims[-1])
        self.critic_layer = nn.Linear(network_dims[-2], 1)
        
        if device is not None and not isinstance(device, torch.device):
            raise TypeError(f"'device' must be a torch.device, got {type(device).__name__}")
        self.device = device if device is not None else torch.device('cpu')
        self.to(self.device)

    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        for layer in self.layers:
            x = layer(x)
        action = F.softmax(self.actor_layer(x), dim=-1)
        state_value = self.critic_layer(x)
        return action, state_value