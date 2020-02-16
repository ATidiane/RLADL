""" FDMS - TME4, Policy Gradient

Auteurs :
* BIZZOZZERO Nicolas
* ADOUM Robert

Liens utiles :
* https://danieltakeshi.github.io/2018/06/28/a2c-a3c
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


class ModuleLamprier(nn.Module):
    def __init__(self, dim_input, dim_output, layers=[], activation=torch.tanh,
                 func_output=None):
        super(ModuleLamprier, self).__init__()
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

        self.pi = ModuleLamprier(dim_input, dim_output, layers=layers,
                                 func_output=nn.Softmax(-1))
        self.V = ModuleLamprier(dim_input, dim_output, layers=layers)
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
            y_target = reward + (self.gamma * self.V(observation))
        loss_V = self.loss_V(y_pred, y_target)
        loss_V.backward()
        self.optim_V.step()

        # Etape 3
        # Evaluation de A
        self.optim_pi.zero_grad()
        A = (y_target - y_pred[action].detach()).detach()

        # Etape 4
        # Approximation de J
        loss_pi = self.loss_pi(
            self.pi(old_state).view(1, self.dim_output),
            torch.tensor([action])
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

        action = torch.Tensor([b.action for b in batch])
        reward = torch.Tensor([b.reward for b in batch])
        observation = torch.Tensor(
            [np.array(list(b.next_state)) for b in batch])
        old_state = torch.Tensor([b.state.detach().numpy() for b in batch])

        # Learn with the batched transitions
        super(A2CBatch, self).learn(old_state, action, observation, reward)


def set_all_seeds(env, seed):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def run_agent_on_environment(agent, env, envx, max_episode, iter_print,
                             iter_show):
    rewards = [0]
    for episode in range(1, max_episode + 1):
        state = torch.tensor(env.reset(), dtype=torch.float32,
                             requires_grad=True)
        done = False
        while not done:
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            reward = torch.FloatTensor([reward])
            next_state = torch.FloatTensor(next_state)

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


def init_environment(env_id, path_dir_output="{env_id}-results", seed=None):
    # Format the path of the output directory
    path_dir_output = path_dir_output.format(env_id=env_id)

    envx = gym.make(env_id)
    env = envx
    env = wrappers.Monitor(envx, directory=path_dir_output,
                           force=True, video_callable=False)
    if seed is not None:
        set_all_seeds(env, seed)

    return env, envx


def main():
    # Initialisation de l'environnement
    env_id = 'CartPole-v1'
    env, envx = init_environment(env_id, seed=0)

    # Initialisation de l'agent
    dim_input = env.observation_space.shape[0]
    dim_output = env.action_space.n
    gamma = 0.99
    alpha = 0.7
    layers = [200]
    lr_V = 0.001
    lr_pi = 0.001
    memory_capacity = 1000
    batch_size = 200
    max_episode = 10
    iter_print = 10
    iter_show = 100
    # agent = A2C(dim_input=dim_input, dim_output=dim_output, gamma=gamma,
    #             alpha=alpha, layers=layers, lr_V=lr_V, lr_pi=lr_pi)
    agent = A2CBatch(dim_input=dim_input, dim_output=dim_output, gamma=gamma,
                     alpha=alpha, layers=layers, lr_V=lr_V, lr_pi=lr_pi,
                     memory_capacity=memory_capacity, batch_size=batch_size)

    run_agent_on_environment(agent, env, envx,
                             max_episode=max_episode,
                             iter_print=iter_print,
                             iter_show=iter_show)


if __name__ == '__main__':
    main()
