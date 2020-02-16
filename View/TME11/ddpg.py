""" FDMS - TME5, DDPG

Auteurs :
* BIZZOZZERO Nicolas
* ADOUM Robert

Liens utiles :
* https://arxiv.org/pdf/1509.02971.pdf
* http://gym.openai.com/envs/LunarLanderContinuous-v2/
* https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
* https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html?fbclid=IwAR1LWm8PSpHHvUOAVVfg7oQ04br6tHQrSzCD0m4qwzrXEzTI-v6A9P-sKBg
"""

import random
import math
from itertools import count
from collections import namedtuple

import gym
from gym import wrappers, logger
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# import envs


matplotlib.use("TkAgg")
SEED_ENVIRONMENT = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if batch_size < len(self):
            return random.sample(self.memory, batch_size)
        return random.sample(self.memory, len(self))

    def __len__(self):
        return len(self.memory)


class QModule(nn.Module):
    def __init__(self, dim_actions, dim_states, dim_hidden_Q,
                 activation=torch.tanh):
        super(QModule, self).__init__()
        self.activation = activation
        self.Q_left = nn.Linear(dim_actions, dim_hidden_Q)
        self.Q_right = nn.Linear(dim_states, dim_hidden_Q)
        self.Q = nn.Linear(dim_hidden_Q, 1)

    def forward(self, state, action):
        return self.Q(self.activation(self.Q_left(action)) +
                      self.activation(self.Q_right(state)))


class MuModule(nn.Module):
    def __init__(self, dim_actions, dim_states, dim_hidden_mu):
        super(MuModule, self).__init__()
        self.mu = nn.Sequential(
            nn.Linear(dim_states, dim_hidden_mu),
            nn.Linear(dim_hidden_mu, dim_actions)
        )

    def forward(self, state):
        return self.mu(state)


class DDPG(object):
    """
    Réseau Q : Critique
    Réseau mu : Acteur
    """

    def __init__(self, dim_actions, dim_states, dim_hidden_Q, dim_hidden_mu,
                 gamma, batch_size=10, memory_size=100000, lr_Q=10e-3,
                 lr_mu=10e-2):
        self.dim_actions = dim_actions
        self.dim_states = dim_states
        self.dim_hidden_Q = dim_hidden_Q
        self.dim_hidden_mu = dim_hidden_mu
        self.gamma = gamma
        self.batch_size = batch_size

        self.replay_buffer = ReplayMemory(memory_size)

        self.Q = QModule(dim_actions, dim_states, dim_hidden_Q)
        self.Q_prime = QModule(dim_actions, dim_states, dim_hidden_Q)
        self.Q_prime.load_state_dict(self.Q.state_dict())
        self.Q_prime.eval()

        self.mu = MuModule(dim_actions, dim_states, dim_hidden_mu)
        self.mu_prime = MuModule(dim_actions, dim_states, dim_hidden_mu)
        self.mu_prime.load_state_dict(self.mu.state_dict())
        self.mu_prime.eval()

        self.optim_Q = optim.Adam(self.Q.parameters(), lr=lr_Q)
        self.optim_mu = optim.Adam(self.mu.parameters(), lr=lr_mu)
        self.loss_Q = nn.MSELoss()

    def act(self, observation):
        noise = torch.Tensor(np.random.uniform(-1, 1, 2))
        return torch.clamp(self.mu(observation) + noise,
                           min=-1, max=1).detach().numpy()

    def learn(self, old_state, action, observation, reward):
        self.replay_buffer.push(old_state, action, observation, reward)
        batch = self.replay_buffer.sample(self.batch_size)

        actions = torch.Tensor([b.action for b in batch])
        rewards = torch.Tensor([b.reward for b in batch])
        states_next = torch.Tensor([b.next_state for b in batch])
        states_old = torch.Tensor([b.state.detach().numpy() for b in batch])

        y_pred = rewards + \
            (self.gamma * self.Q_prime(states_next, self.mu_prime(states_next)))

        self.optim_Q.zero_grad()
        loss = self.loss_Q(y_pred, self.Q(states_old, actions))
        loss.backward()
        self.optim_Q.step()

        self.optim_mu.zero_grad()
        action_pred = self.mu(states_old)
        Q = self.Q(states_old, action_pred)
        (-Q).sum().backward()
        self.optim_mu.step()

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

    def MAJ_parameters(self):
        self.Q_prime.load_state_dict(self.Q.state_dict())
        self.Q_prime.eval()

        self.mu_prime.load_state_dict(self.mu.state_dict())
        self.mu_prime.eval()


def main():
    # Chargement de l'environnement
    env_id = 'LunarLanderContinuous-v2'
    outdir = 'output'
    envx = gym.make(env_id)
    env = envx
    env = wrappers.Monitor(envx, directory=outdir,
                           force=True, video_callable=False)

    # Définition des graines aléatoires
    random.seed(SEED_ENVIRONMENT)
    np.random.seed(SEED_ENVIRONMENT)
    env.seed(SEED_ENVIRONMENT)
    torch.manual_seed(SEED_ENVIRONMENT)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED_ENVIRONMENT)
        torch.cuda.manual_seed_all(SEED_ENVIRONMENT)

    # On récupère les dimensions des espaces d'action et d'observation
    if isinstance(env.action_space, gym.spaces.box.Box):
        dim_actions = env.action_space.shape[0]
    else:
        dim_actions = env.action_space.n
    dim_states = env.observation_space.shape[0]

    # Apprentissage
    dim_hidden_Q = 100
    dim_hidden_mu = 100
    gamma = 0.99  # 0.7
    batch_size = 128
    memory_size = 100000
    lr_Q = 10e-2
    lr_mu = 10e-2
    max_episode = 10000
    step_update_parameters = 10
    iter_print = 10
    iter_show = 100
    agent = DDPG(dim_actions=dim_actions, dim_states=dim_states,
                 dim_hidden_Q=dim_hidden_Q, dim_hidden_mu=dim_hidden_mu,
                 gamma=gamma, batch_size=batch_size, memory_size=memory_size,
                 lr_Q=lr_Q, lr_mu=lr_mu)

    rewards = [0]
    for episode in range(1, max_episode + 1):
        state = torch.tensor(env.reset(), dtype=torch.float32,
                             requires_grad=True)
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, next_state, reward)

            rewards[-1] += reward
        rewards.append(0)

        if episode % iter_print == 0:
            print("Mean of", iter_print, "last rewards for episode", episode,
                  ":", round(np.mean(rewards), 3),
                  "(std {})".format(round(np.std(rewards), 3)))
            rewards = [0]

        if episode % iter_show == 0:
            agent.show(env, envx)

        # TODO: Mettre dans la classe
        if episode % step_update_parameters == 0:
            agent.MAJ_parameters()


if __name__ == '__main__':
    main()
