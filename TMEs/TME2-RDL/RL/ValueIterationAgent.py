import copy

import gym
import matplotlib
import numpy as np
from gym import logger, wrappers

import gridworld

matplotlib.use("TkAgg")


def valueIteration(mdp, epsilon=0.01):
    """ Returns the Value Iteration
    """

    # Initialize Vo randomly
    Vi = dict(zip(mdp.keys(), np.random.rand(len(mdp))))

    # Set gamma to 1 in this case
    gamma = 1

    # For each action, compute V and then take the max
    while True:
        Viplus1 = dict(zip(mdp.keys(), np.zeros(len(mdp))))
        for state_key, state_value in mdp.items():
            actions_value = []
            for action in state_value.values():
                v = 0
                for i, possible_transition in enumerate(action):
                    proba, s_prime, reward, done = possible_transition
                    if s_prime not in Vi.keys():
                        continue
                    v += proba * (reward + (gamma * Vi[s_prime]))

                actions_value.append(v)

            Viplus1[state_key] = max(actions_value)

        diff_v = np.array(list(Viplus1.values())) - np.array(list(Vi.values()))
        if np.linalg.norm(diff_v) <= epsilon:
            Vi = Viplus1
            break

        Vi = Viplus1

        pi = dict()
        for state_key, state_value in mdp.items():
            actions = dict()
            for action_key, action_value in state_value.items():
                v = 0
                for i, possible_transition in enumerate(action_value):
                    proba, s_prime, reward, done = possible_transition
                    if s_prime not in Vi.keys():
                        continue
                    v += proba * (reward + (gamma * Vi[s_prime]))

                actions[action_key] = v

            pi[state_key] = max(actions, key=lambda key: actions[key])
            
    return pi


def policyIteration(mdp):
    """ Returns the policy Iteration
    """
    pass


class Agent(object):

    def __init__(self, action_space, epsilon, gamma=1):
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma

    def act(self, observation, reward, done):
        # On a rien dans cette fonction exepté par exemple return pi[s]
        return self.action_space.sample()

    # On fait tout dans une fonction fit ou dans init.


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
    agent = Agent(env.action_space)

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