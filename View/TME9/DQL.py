import sys
import argparse
import copy
import json
import random
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import gym
from gym import wrappers, logger
# import envs

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F

from collections import deque
import random

from tqdm import tqdm


class Q_module(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(Q_module, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))

    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.nn.functional.leaky_relu(x)
            x = self.layers[i](x)
        return x


class DQL_ER(object):
    def __init__(self, inSize, outSize, env, envx, layers=[200],
                 minibatch_size=200, eps=0.5, gamma=0.99, lr=0.001, C=4,
                 size_deque=300):
        self.D = deque(maxlen=size_deque)
        self.action_space = env.action_space
        self.Q = Q_module(inSize, outSize, layers)
        self.Q_hat = Q_module(inSize, outSize, layers)
        self.Q_hat.load_state_dict(self.Q.state_dict())
        self.optimizer = optim.Adam(self.Q.parameters(), lr=lr)
        self.loss = nn.MSELoss()  # nn.SmoothL1Loss()

        self.eps = eps
        self.C = C
        self.envx = envx
        self.gamma = gamma
        self.minibatch_size = minibatch_size

    def fit(self, M, env, graph=True, verbose=False):
        # reward_list = []
        # loss_list = []
        v = 0
        c = 1
        for m in range(M):
            state = env.reset()
            done = False
            r = 0
            while not done:
                # With probability eps select a random action
                action = self.select_action(state)

                # Execute action and observe reward r and next state
                next_state, reward, done, _ = env.step(action)
                r += reward

                # Si done = True alors 1 - done = 0 et on retrouve bien que
                # target = reward
                target = reward + (1 - done) * self.gamma * \
                    self.get_max(next_state)
                self.D.append([state, action, reward, next_state, target])

                # Gradient descent
                if c >= self.minibatch_size:
                    minibatch = random.sample(self.D, self.minibatch_size)
                    y = torch.tensor([mb[4] for mb in minibatch],
                                     dtype=torch.float32)
                    x = torch.tensor([mb[0] for mb in minibatch],
                                     dtype=torch.float32, requires_grad=True)
                    self.optimizer.zero_grad()
                    y_pred = self.Q(x)
                    loss = self.loss(torch.max(y_pred, 1)[0], y)
                    loss.backward()
                    self.optimizer.step()

                    # reward_list.append(r)
                    # loss_list.append(loss.detach())

                    if c % self.C == 0:
                        self.Q_hat.load_state_dict(self.Q.state_dict())
                state = next_state
                c += 1

            if verbose:
                print("itération:", v, "reward:", r)

            v += 1
            self.update_eps()

            # if c % 100 == 0:
            #    self.test(self.envx, 1, verbose=True, graph=False)

        if graph:
            x = [i for i in range(1, M + 1)]
            plt.plot(x, reward_list)
            plt.xlabel("Nombre de parties")
            plt.ylabel("Score")
            plt.show()
            plt.plot(x, loss_list)
            plt.xlabel("Nombre de parties")
            plt.ylabel("Loss en fin de partie")
            plt.show()

    def select_action(self, state):
        if np.random.uniform() < self.eps:
            return self.action_space.sample()

        with torch.no_grad():
            pred = self.Q(torch.tensor(state, dtype=torch.float32))
        return torch.argmax(pred).numpy()

    def act(self, state):
        with torch.no_grad():
            pred = self.Q(torch.tensor(state, dtype=torch.float32))
        return torch.argmax(pred).numpy()

    def get_max(self, next_state):
        with torch.no_grad():
            pred = self.Q_hat(torch.tensor(next_state, dtype=torch.float32))
        maxi = torch.max(pred.detach())
        return maxi

    def update_eps(self):
        self.eps -= 0.005

    def test(self, envx, T, graph=True, verbose=False, demo_jeu=False):
        reward_list = []
        for i in range(T):
            if demo_jeu:
                envx.render(1)
            state = env.reset()
            done = False
            r = 0
            while not done:
                action = self.act(state)
                if demo_jeu:
                    envx.render()
                next_state, reward, done, _ = env.step(action)
                r += reward
                state = next_state

            if verbose:
                print(r)
            reward_list.append(r)
        if graph:
            plt.plot([i for i in range(1, T + 1)], reward_list)
            plt.xlabel("Nombre de parties")
            plt.ylabel("Score")
            plt.show()


def set_all_seeds(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    inSize = 4
    outSize = 2
    M = 800

    env_id = 'CartPole-v1'
    outdir = 'output'
    envx = gym.make(env_id)
    envx.verbose = True
    env = envx
    env = wrappers.Monitor(envx, directory=outdir, force=True,
                           video_callable=False)
    set_all_seeds(env, seed=1)

    dql = DQL_ER(inSize, outSize, env, envx)
    dql.fit(M, env, verbose=True, graph=False)
    # dql.test(envx, 100, verbose=True, graph=False)
