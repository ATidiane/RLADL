import matplotlib

matplotlib.use("TkAgg")
import pandas as pd
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import argparse
from randomAgent import RandomAgent
from valueIteration import ValueIteration
from policyIteration import PolicyIteration

def main(args):
    env = gym.make("gridworld-v0")
    env.setPlan("gridworldPlans/plan{}.txt".format(args.plan), {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1})
    env.verbose = True
    statedic, mdp = env.getMDP()  # recupere le mdp : statedic

    columns = ['Episode', 'rsum', 'actions']
    pd_exp = pd.DataFrame(columns=columns)
    
    # Execution avec un Agent
    if args.agent == 'value':
        agent = ValueIteration(env.action_space, statedic, mdp)
    elif args.agent == 'policy':
        agent = PolicyIteration(env.action_space, statedic, mdp)
    else:
        agent = RandomAgent(env.action_space)

    # Faire un fichier de log sur plusieurs scenarios
    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)

    episode_count = args.episodes
    reward = 0
    done = False
    rsum = 0
    FPS = args.fps
    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
            env.render(mode="human")
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
                pd_exp = pd_exp.append(pd.DataFrame([i, rsum, j], index=columns).T,
                                       ignore_index=True)
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break

    pd_exp.to_csv('./csv_results/{0}_{1}.csv'.format(args.agent, args.plan), index=False)
    print("done")
    env.close()
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', metavar='N', type=int, default=10000,
                          help='number of episodes')
    parser.add_argument('--agent', type=str, default="policy",
                          help="the agent to use, value or policy")
    parser.add_argument('--fps', type=float, default=0.0001,
                          help='frame per second')
    parser.add_argument('--plan', type=int, default=0,
                        help='GridWordplan number from 0 to 10')
    args = parser.parse_args()
    
    main(args)
