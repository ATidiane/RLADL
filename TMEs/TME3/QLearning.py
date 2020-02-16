import gridworld
import numpy as np
import copy


class QLearning:
    def __init__(self, action_space, epsilon=0.1, alpha=0.1, gamma=0.7, eligibility=False):
        self.action_space = action_space
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.eligibility = eligibility
        self.states = dict()        
        self.Q = dict()
        self.lastobs = None
        self.lasta = None
        
            
    def act(self, observation, reward, done):
        """ Returns an action to make

        :param observation: observation of the world
        :param reward: the reward
        :param done: the end of the episode
        :returns: a number
        :rtype: Int

        """

        if self.elgibility:
            return self.act_with_eligibility(observation, reward, done)
        else:
            return self.act_without_eligibility(observation, reward, done)

    def act_without_eligibility(self, observation, reward, done):
        observation = gridworld.gridworld_env.GridWordEnv.state2str(observation)
        if observation not in list(self.Q.keys()):
            self.Q[observation] = np.zeros(self.action_space.n)

        if self.lastobs is not None:
            q_st_at = self.Q[self.lastobs][self.lasta]
            self.Q[self.lastobs][self.lasta] = q_st_at + self.epsilon * (
                reward + self.gamma * (
                    self.Q[observation][np.argmax(self.Q[observation])] - q_st_at))

        if np.random.random() < self.epsilon:
            act = self.action_space.sample()
        else:
            act = np.argmax(self.Q[observation])

        self.lasta = int(act)
        self.lastobs = observation

        return act

    def act_with_eligibility(self, observation, reward, done):
        pass
