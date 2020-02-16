""" FDMS - TME6, DAGGER

Auteurs :
* BIZZOZZERO Nicolas
* ADOUM Robert
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


class Dagger(object):
    def __init__(self, dim_actions, dim_states, dim_hidden, lr=10e-3,
                 beta=0.01):
        self.dim_actions = dim_actions
        self.dim_states = dim_states
        self.dim_hidden = dim_hidden
        self.beta = beta

        self.module = torch.nn.Sequential(
            nn.Linear(dim_states, dim_hidden),
            nn.Linear(dim_hidden, dim_actions)
        )
        self.optim = optim.Adam(self.module.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

    def act(self, observation, expert_policy):
        pass

    def learn(self, old_state, action, observation, reward, expert_policy):
        pass

    def show(self, env, envx):
        envx.render(1)
        state = torch.tensor(env.reset(), dtype=torch.float32,
                             requires_grad=False)
        done = False
        while not done:
            action = lunar_lander_heuristic(env, state)
            # action = self.act(state)
            envx.render()
            state, reward, done, _ = env.step(action)
            state = torch.tensor(state, dtype=torch.float32)


def main():
    # Chargement de l'environnement
    env_id = 'LunarLander-v2'
    outdir = 'LunarLander-v2/ddpg_results'
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

    # Apprentissage
    dim_actions = 2
    dim_states = 2
    dim_hidden = 1024
    lr = 10e-2
    beta = 0.01
    step_update_parameters = 1
    max_episode = 1000
    iter_print = 100
    iter_show = 100
    agent = Dagger(dim_actions=dim_actions, dim_states=dim_states, dim_hidden=dim_hidden,
                   lr=lr, beta=beta)

    rewards = [0]
    for episode in range(1, max_episode + 1):
        state = torch.tensor(env.reset(), dtype=torch.float32,
                             requires_grad=False)
        done = False
        while not done:
            # pi_i = beta * expert_policy(state) + (1 - beta) * agent.act(state)

            # action = agent.act(state)
            action = lunar_lander_heuristic(env, state)
            next_state, reward, done, _ = env.step(action)
            # agent.learn(state, action, next_state, reward)
            
            rewards[-1] += reward
        rewards.append(0)

        if episode % iter_print == 0:
            print("Mean of", iter_print, "last rewards for episode", episode,
                  ":", round(np.mean(rewards), 3),
                  "(std {})".format(round(np.std(rewards), 3)))
            rewards = [0]

        if episode % iter_show == 0:
            agent.show(env, envx)


def lunar_lander_heuristic(env, s):
    """
    Source :
    https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py#L364
    """
    # Heuristic for:
    # 1. Testing. 
    # 2. Demonstration rollout.
    angle_targ = s[0] * 0.5 + s[2] * 1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
    #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
    #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    if s[6] or s[7]: # legs have contact
        angle_todo = 0
        hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
    elif angle_todo < -0.05: a = 3
    elif angle_todo > +0.05: a = 1
    return a



if __name__ == '__main__':
    main()
