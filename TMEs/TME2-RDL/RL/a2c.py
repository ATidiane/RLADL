""" FDMS - TME5, Policy Gradient

Author : BALDÉ Ahmed

"""

import random
from collections import namedtuple

import gym
from gym import wrappers
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# import envs


matplotlib.use("TkAgg")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class a2c(object):
    def __init__(self, action_space, gamma, alpha, layers, lr_V=0.001, lr_pi=0.001):
        self.action_space = action_space
        self.gamma = gamma
        self.alpha = alpha
        # Coût de Huber
        self.pi = None
        self.V = None
        self.loss_V = nn.SmoothL1Loss()
        self.loss_pi = nn.CrossEntropyLoss()
        self.optim_V = optim.Adam(self.V.parameters(), lr=lr_V)
        self.optim_pi = optim.Adam(self.pi.parameters(), lr=lr_pi)



    def act(self, observation, reward, done):
        with torch.no_grad():
            distribution = torch.distributions.categorical.Categorical(
                probs = self.pi(observation)
            )
            return self.distribution.sample().item()

    def fit(self, observation, reward, done):
        # Step 2 : Update V
        self.optim_V.zero_grad()
        
        with torch.no_grad():
            y_target
