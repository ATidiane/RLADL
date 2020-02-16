
import matplotlib
matplotlib.use("TkAgg")
import copy
import gym
import numpy as np
from gym import logger, wrappers

import gridworld




class QLearning(object):
    """ Update le Q et l'env interagit strictement avec le main.
    """
    def __init__(self, action_space, epsilon=0.01, gamma=0.99, eligibility=False):
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.lasta = None
        self.lastobs = None
        self.Q = dict()
        self.eligibility = eligibility

    def act(self, observation, reward, done):
        if self.eligibility:
            return self.act_with_eligibility(observation, reward, done)
        return self.act_without_eligibility(observation, reward, done)


    def act_without_eligibility(self, observation, reward, done):
        """ Algorithme Q-Learning
        """

        observation = gridworld.gridworld_env.GridworldEnv.state2str(observation)
        if observation not in list(self.Q.keys()):
            self.Q[observation] = np.zeros(self.action_space.n)

        if self.lastobs is not None:
            q_st_at = self.Q[self.lastobs][self.lasta] 
            self.Q[self.lastobs][self.lasta] = q_st_at + self.epsilon * (
                reward + self.gamma * (
                    self.Q[observation][np.argmax(self.Q[observation])] - q_st_at))

        if np.random.uniform() < self.epsilon:
            act = self.action_space.sample()            
        else:
            act = np.argmax(self.Q[observation])
            
        self.lasta = int(act)
        self.lastobs = observation

        return act

    def act_with_eligibility(self, observation, reward, done):
        """ Algorithme Q-Learning with eligibility traces
        """

        #TODO

        if observation not in self.states.keys():
            self.Q[observation] = np.zeros(self.action_space.n)

        if np.random.uniform() < epsilon:
            act = self.action_space.sample()
            self.lasta = act
            self.lastobs = observation
            return act
        else:
            ind_max = np.argmax(Q[observation])
            return Q[observation][ind_max]

        for state in Q.keys():
            tmp_q_state = Q[state]
            for i in range(self.action_space.n):
                Q[state][i] = None

        return self.action_space.sample()

    


if __name__ == "__main__":
    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random
    print(env.action_space)  # Quelles sont les actions possibles
    print(env.step(1))  # faire action 1 et retourne l'observation, le reward, et un done un booleen (jeu fini ou pas)
    env.render()  # permet de visualiser la grille du jeu 
    env.render(mode="human") #visualisation sur la console
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic
    print("Nombre d'etats : ",len(statedic))  # nombre d'etats ,statedic : etat-> numero de l'etat
    state, transitions = list(mdp.items())[0]
    print(state)  # un etat du mdp
    print(transitions)  # dictionnaire des transitions pour l'etat :  {action-> [proba,etat,reward,done]}

    # Execution avec un Agent
    agent = QLearning(env.action_space, epsilon=0.01, gamma=0.99, eligibility=False)

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan0.txt", {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = 1000
    reward = 0
    done = False
    rsum = 0
    FPS = 0.0001
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    print("done")
    env.close()