import random
from collections import namedtuple

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gym import wrappers

# import envs


matplotlib.use("TkAgg")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """ Saves a transition. """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """ Return a sample of length `batch_size` from the memory. """
        if batch_size < len(self):
            return random.sample(self.memory, batch_size)
        return random.sample(self.memory, len(self))

    def __len__(self):
        return len(self.memory)


class NN(nn.Module):
    def __init__(self, dim_input, dim_output, layers=[], activation=torch.tanh,
                 func_output=None):
        super(NN, self).__init__()
        self.activation = activation
        self.func_output = func_output
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(dim_input, x))
            dim_input = x
        self.layers.append(nn.Linear(dim_input, dim_output))

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = self.activation(x)
            x = self.layers[i](x)

        if self.func_output is not None:
            return self.func_output(x)
        else:
            return x


class A2C(object):
    def __init__(self, dim_input, dim_output, gamma, alpha, layers,
                 lr_V=0.001, lr_pi=0.001):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.gamma = gamma
        self.alpha = alpha

        self.pi = NN(dim_input, dim_output, layers=layers,
                     func_output=nn.Softmax(-1))
        self.V = NN(dim_input, dim_output, layers=layers)
        self.optim_V = optim.Adam(self.V.parameters(), lr=lr_V)
        self.optim_pi = optim.Adam(self.pi.parameters(), lr=lr_pi)
        self.loss_V = nn.SmoothL1Loss()
        self.loss_pi = nn.CrossEntropyLoss()

    def act(self, observation):
        with torch.no_grad():
            distribution = torch.distributions.Categorical(
                probs=self.pi(observation)
            )
            return distribution.sample().item()

    def learn(self, old_state, action, observation, reward):
        # Etape 2
        # On met à jour V
        self.optim_V.zero_grad()
        y_pred = self.V(old_state)
        with torch.no_grad():
            # print("reward", reward.view(-1, 1))
            # print("obs", self.V(observation))
            y_target = reward.view(-1, 1) + (self.gamma * self.V(observation))
        loss_V = self.loss_V(y_pred, y_target)
        loss_V.backward()
        self.optim_V.step()

        # Etape 3
        # Evaluation de A

        self.optim_pi.zero_grad()
        # print("y_pred", y_pred)
        # print("y_target", y_target)
        # A = (y_target - y_pred[action].detach().view(-1, 1)).detach()

        A = (y_target - y_pred[range(len(y_pred)),
                               action].detach().view(-1, 1)).detach()

        # Etape 4
        # Approximation de J
        # print("old_state", old_state)
        # print(
        #     "old_pi",
        #     self.pi(old_state).view(
        #         old_state.size(0),
        #         self.dim_output))
        # print("action", action)
        # print(old_state.size(0))
        loss_pi = self.loss_pi(
            self.pi(old_state).view(old_state.size(0), self.dim_output),
            action
        ) * A

        # Etape 5
        # Mise à jour des paramètres de PI
        loss_pi.sum().backward()
        self.optim_pi.step()

    def show(self, env, envx):
        envx.render(1)
        state = torch.tensor(env.reset(), dtype=torch.float32,
                             requires_grad=True)
        done = False
        while not done:
            action = self.act(state)
            envx.render()
            state, reward, done, _ = env.step(action)
            state = torch.tensor(state, dtype=torch.float32)


class A2CBatch(A2C):
    def __init__(self, dim_input, dim_output, gamma, alpha, layers,
                 lr_V=0.001, lr_pi=0.001, memory_capacity=200, batch_size=200):
        super(A2CBatch, self).__init__(dim_input, dim_output, gamma, alpha,
                                       layers, lr_V, lr_pi)

        self.memory = ReplayMemory(memory_capacity)
        self.batch_size = batch_size

    def learn(self, old_state, action, observation, reward):
        # Store the new transition
        self.memory.push(old_state, action, observation, reward)

        # Retrieve passed transitions
        batch = self.memory.sample(self.batch_size)

        action = torch.tensor([b.action for b in batch],
                              dtype=torch.long)
        reward = torch.tensor([b.reward for b in batch])
        observation = torch.tensor(
            [np.array(list(b.next_state)) for b in batch])
        old_state = torch.tensor([b.state.detach().numpy() for b in batch])

        # Learn with the batched transitions
        super(A2CBatch, self).learn(old_state, action, observation, reward)
