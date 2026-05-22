import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# copy from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_rnd_envpool.py
# ALGO LOGIC: initialize agent here:
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, use_layer_init):
        super(Model, self).__init__()
        self.conv1 = layer_init(nn.Conv2d(input_dim[0], 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(input_dim[0], 32, 3, stride=2, padding=1)
        self.conv2 = layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = layer_init(nn.Linear(1152, 512)) if use_layer_init else nn.Linear(1152, 512)
        self.in_critic = layer_init(nn.Linear(512, 1)) if use_layer_init else nn.Linear(512, 1)
        self.ex_critic = layer_init(nn.Linear(512, 1)) if use_layer_init else nn.Linear(512, 1)
        self.actor_linear = layer_init(nn.Linear(512, output_dim)) if use_layer_init else nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x/255.))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear(x))
        return self.actor_linear(x), self.ex_critic(x), self.in_critic(x)

class Random_Encoder_Model(nn.Module):
    def __init__(self, embedding_dim, use_layer_init):
        super(Random_Encoder_Model, self).__init__()

        self.encoder1 = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(1152, 150)) if use_layer_init else nn.Linear(1152, 150),
            nn.ReLU(),
            layer_init(nn.Linear(150, embedding_dim)) if use_layer_init else nn.Linear(150, embedding_dim),
        )

        self.encoder2 = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(1152, 150)) if use_layer_init else nn.Linear(1152, 150),
            nn.ReLU(),
            layer_init(nn.Linear(150, embedding_dim)) if use_layer_init else nn.Linear(150, embedding_dim),
        )

        self.encoder3 = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 3, stride=2, padding=1)) if use_layer_init else nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(1152, 150)) if use_layer_init else nn.Linear(1152, 150),
            nn.ReLU(),
            layer_init(nn.Linear(150, embedding_dim)) if use_layer_init else nn.Linear(150, embedding_dim),
        )

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        x = x/255.
        x1 = self.encoder1(x)
        x2 = self.encoder2(x)
        x3 = self.encoder3(x)
        return (x1+x2+x3)/3.