import gridworld
import numpy as np
import copy


class ValueIteration():
    def __init__(self, action_space, statedic, mdp, epsilon=1e5, gamma=0.9):
        """ Value Iteration algorithm

        :param action_space: discrete(4)
        :param statedic: all states 
        :param mdp: P
        :param epsilon: epsilon
        :param gamma: gamma
        :returns: None
        :rtype: Void

        """
        self.states = [s for s in statedic]
        self.non_term_states = [s for s in mdp]
        self.P = mdp
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma

        self.pi = {s: None for s in self.states}

        Vi = dict(zip(self.states, np.random.random(len(self.states))))
        Vi_plus = dict(zip(self.states, np.zeros(len(self.states))))

        for s in self.non_term_states:
            Vi_plus[s] = np.max([np.sum([proba * (reward + self.gamma * Vi[s_prime])
                          for (proba, s_prime, reward, boolean) in self.P[s][a]])
                                 for a in range(self.action_space.n)])

        while np.linalg.norm(np.array(list(Vi.values())) - np.array(list(Vi_plus.values())), ord=np.inf) > self.epsilon:
            Vi = copy.deepcopy(Vi_plus)
            for s in self.non_term_states:
                Vi_plus[s] = np.max([np.sum([proba * (reward + self.gamma * Vi[s_prime])
                          for (proba, s_prime, reward, boolean) in self.P[s][a]])
                                     for a in range(self.action_space.n)])

        for s in self.non_term_states:
            self.pi[s] = np.argmax([np.sum([proba * (reward + self.gamma * Vi[s_prime])
                          for (proba, s_prime, reward, boolean) in self.P[s][a]])
                                    for a in range(self.action_space.n)])

        
    def act(self, observation, reward, done):
        """ Returns an action to make

        :param observation: observation of the world
        :param reward: the reward
        :param done: the end of the episode
        :returns: a number
        :rtype: Int

        """
        return self.pi[gridworld.GridworldEnv.state2str(observation)]
