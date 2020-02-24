import copy
import random
import sys

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import logger, wrappers

import gridworld


class QLearning(object):

    def __init__(self, env, learning_rate, discount, epsilon=0.1):
        """Q-Learning

        :param env:
        :param learning_rate:
        :param discount:
        :param epsilon:
        :returns:
        :rtype:

        """
        self.env = env
        self.action_space = env.action_space
        self.Q = {}
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.lastobs = None
        self.lasta = None

    def act(self, observation, reward, done):
        """Return an action giving the following

        :param observation:
        :param reward:
        :param done:
        :returns:
        :rtype:

        """

        state = self.env.state2str(observation)
        self.obs = state
        self.Q.setdefault(state, [0, 0, 0, 0])
        self.reward = reward
        if np.random.random() < 1 - self.epsilon:
            action = np.argmax(self.Q[self.obs])
        else:
            action = np.random.randint(self.action_space.n)

        self.update_q(action)
        return action

    def update_q(self, action):
        if not self.lastobs is None:
            st = self.lastobs
            st1 = self.obs
            self.Q[st][self.lasta] += self.learning_rate * (self.reward +
                                                            self.discount * np.max(self.Q[st1]) - self.Q[st][self.lasta])

        self.lastobs = self.obs
        self.lasta = action


class Sarsa(object):

    def __init__(self, env, learning_rate, discount, epsilon=0.1):
        """Sarsa

        :param env:
        :param learning_rate:
        :param discount:
        :param epsilon:
        :returns:
        :rtype:

        """
        self.env = env
        self.action_space = env.action_space
        self.Q = {}
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.lastobs = None
        self.lasta = None

    def act(self, observation, reward, done):
        """Return an action giving the following

        :param observation:
        :param reward:
        :param done:
        :returns:
        :rtype:

        """

        state = self.env.state2str(observation)
        self.obs = state
        self.Q.setdefault(state, [0, 0, 0, 0])
        self.reward = reward
        if np.random.random() < 1 - self.epsilon:
            action = np.argmax(self.Q[self.obs])
        else:
            action = np.random.randint(self.action_space.n)

        self.update_q(action)
        return action

    def update_q(self, action):
        if not self.lastobs is None:
            st = self.lastobs
            st1 = self.obs
            self.Q[st][self.lasta] += self.learning_rate * (self.reward +
                                                            self.discount * self.Q[st1][action] - self.Q[st][self.lasta])

        self.lastobs = self.obs
        self.lasta = action


class DynaQ(object):

    def __init__(self, env, learning_rate, discount,
                 planning_step, epsilon=0.1):
        """Dyna-Q

        :param env:
        :param learning_rate:
        :param discount:
        :param planning_step:
        :param epsilon:
        :returns:
        :rtype:

        """
        self.env = env
        self.action_space = env.action_space
        self.Q = {}
        self.model = {}
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount = discount
        self.lastobs = None
        self.lasta = None
        self.planning_step = planning_step

    def act(self, observation, reward, done):
        """Return an action, giving the following

        :param observation:
        :param reward:
        :param done:
        :returns:
        :rtype:

        """

        state = self.env.state2str(observation)
        self.obs = state
        self.Q.setdefault(state, [0, 0, 0, 0])
        self.model.setdefault(
            state, [(False, False), (False, False),
                    (False, False), (False, False)])
        self.reward = reward
        if np.random.random() < 1 - self.epsilon:
            action = np.argmax(self.Q[self.obs])
        else:
            action = np.random.randint(self.action_space.n)
        self.update_qm(action)

        return action

    def planning(self):
        for _ in range(min(self.planning_step,
                           len(list(self.model.keys())))):
            st = np.random.choice(list(self.model.keys()))
            action = np.random.randint(self.action_space.n)
            reward, st1 = self.model[st][action]
            cpt = 0
            while reward == False and cpt < 20:
                action = np.random.randint(self.action_space.n)
                reward, st1 = self.model[st][action]
                cpt += 1
            if st1 != False:
                self.Q[st][action] += self.learning_rate * (reward +
                                                            self.discount * np.max(self.Q[st1]) - self.Q[st][action])

    def update_qm(self, action):
        if not self.lastobs is None:
            st = self.lastobs
            st1 = self.obs
            self.Q[st][self.lasta] += self.learning_rate * (self.reward +
                                                            self.discount * np.max(self.Q[st1]) - self.Q[st][self.lasta])
            self.model[st][self.lasta] = (self.reward, st1)
            self.planning()
        self.lastobs = self.obs
        self.lasta = action
