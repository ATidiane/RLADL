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

from utils import set_all_seeds


def run_agent_on_environment(agent, env, envx, max_episode, iter_print,
                             iter_show, name_file):
    rewards = [0]
    mean_rewards, std_rewards = [], []
    for episode in range(1, max_episode + 1):
        state = torch.tensor(env.reset(), dtype=torch.float32,
                             requires_grad=True)
        done = False
        rewards_me = []
        while not done:
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            reward = torch.FloatTensor([reward])
            next_state = torch.FloatTensor(next_state)

            agent.learn(state, action, next_state, reward)
            rewards[-1] += reward
            rewards_me.append(reward)
        rewards.append(0)

        if episode % iter_print == 0:
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)

            print("Mean of", iter_print, "last rewards for episode", episode,
                  ":", mean_reward,
                  "(std {})".format(std_reward))
            rewards = [0]

        if episode % iter_show == 0:
            agent.show(env, envx)

    results = pd.DataFrame(
        np.vstack(
            (mean_rewards, std_rewards)).T, columns=[
                'reward', 'std'])
    results.to_csv("csv_results/{}".format(name_file), index=False)


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
