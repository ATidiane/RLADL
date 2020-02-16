import gridworld
import numpy as np
import copy


class PolicyIteration:
    def __init__(self, action_space, statedic, mdp, epsilon=1e5, gamma=0.99):
        """ Policy Iteration algorithm

        :param action_space: discrete(4)
        :param statedic: all states 
        :param mdp: P
        :param epsilon: epsilon
        :param gamma: gamma
        :returns: None
        :rtype: Void

        """
        self.action_space = action_space
        self.statedic = statedic
        self.states = [s for s in statedic]
        self.P = mdp
        self.epsilon = epsilon
        self.gamma = gamma
        self.non_term_states = [s for s in mdp]

        self.pik = {s:None for s in self.non_term_states}
        pik_plus = {s: self.action_space.sample() for s in self.non_term_states}

        while self.pik != pik_plus:
            self.pik = dict(pik_plus)
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
                pik_plus[s] = np.argmax([np.sum([proba * (reward + self.gamma * Vi[s_prime])
                                                for (proba, s_prime, reward, boolean) in self.P[s][a]])
                                        for a in range(self.action_space.n)])
                
        self.pik = dict(pik_plus)

    def act(self, observation, reward, done):
        """ Returns an action to make

        :param observation: observation of the world
        :param reward: the reward
        :param done: the end of the episode
        :returns: a number
        :rtype: Int

        """        
        return self.pik[gridworld.GridworldEnv.state2str(observation)]
