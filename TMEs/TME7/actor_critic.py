import copy

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import logger, wrappers


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_action, max_action):
        """Actor

        :param state_dim:
        :param action_dim:
        :param min_action:
        :param max_action:
        :returns:
        :rtype:

        """
        super(Actor, self).__init__()

        self.min_action = min_action
        self.action_range = max_action - min_action

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Sigmoid()
        )

    def forward(self, state):
        state = state.float()
        raw_a = self.layers(state)
        a = (raw_a * self.action_range) + self.min_action
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """Critic

        :param state_dim:
        :param action_dim:
        :returns:
        :rtype:

        """
        super(Critic, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, state, action):
        q = self.layers(torch.cat([state, action], 1))
        return q
