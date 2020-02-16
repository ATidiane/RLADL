import matplotlib

matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN, self).__init__()
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

class DQN(object):
    """ Update le Q et l'env interagit strictement avec le main.
    """
    def __init__(self, action_space, epsilon=0.01, gamma=0.99, N=100):
        self.action_space = action_space
        print(self.action_space)
        self.epsilon = epsilon
        self.gamma = gamma
        self.lasta = None
        self.lastobs = None
        self.N = N
        #self.D = [  for _ in N]


    def act(self, observation, reward, done):
        """
        """

        if np.random.uniform() < self.epsilon:
            act = self.action_space.sample()
            print(act)
        else:
            #act = np.argmax(self.Q[observation])
            pass


        observation = gridworld.gridworld_env.GridworldEnv.state2str(observation)
        if observation not in list(self.Q.keys()):
            self.Q[observation] = np.zeros(self.action_space.n)

        if self.lastobs is not None:
            q_st_at = self.Q[self.lastobs][self.lasta] 
            self.Q[self.lastobs][self.lasta] = q_st_at + self.epsilon * (
                reward + self.gamma * (
                    self.Q[observation][np.argmax(self.Q[observation])] - q_st_at))

        
            
        self.lasta = int(act)
        self.lastobs = observation

        return act


if __name__ == '__main__':


    env = gym.make('CartPole-v1')

    # Enregistrement de l'Agent
    agent = DQN(env.action_space, epsilon=1, gamma=0.99)
    outdir = 'cartpole-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.seed(0)

    episode_count = 1000000
    reward = 0
    done = False
    env.verbose = True
    np.random.seed(5)
    rsum = 0

    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render()
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            print(action)
            obs, reward, done, _ = envm.step(action)
            print(obs, reward, done)
            rsum += reward
            j += 1
            if env.verbose:
                env.render()
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()