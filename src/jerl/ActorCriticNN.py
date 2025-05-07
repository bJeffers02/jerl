import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticLNN(nn.Module):
    def __init__(self, network_dims, device):
        super(ActorCriticLNN, self).__init__()

        self.device = device

        self.layers = nn.ModuleList()
        for i in range(len(network_dims) - 2):
            self.layers.append(nn.Linear(network_dims[i], network_dims[i+1]))
        
        self.actor_layer = nn.Linear(network_dims[-2], network_dims[-1])
        self.critic_layer = nn.Linear(network_dims[-2], 1)
        
        self.to(self.device)

    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        action = F.softmax(self.actor_layer(x), dim=-1)
        state_value = self.critic_layer(x)
        return action, state_value