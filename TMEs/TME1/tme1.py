import numpy as np
import pandas as pd

NB_ANNONCEURS = 10


def read_file(file):
    """
    """

    X, y = [], []
    with open(file, 'r') as f:
        for line in f:
            line = f.readline()
            num_article = line.split(':')[0]
            dimensions = list(
                map(lambda x: float(x), line.split(':')[1].split(';')))
            taux = list(map(lambda x: float(x), line.split(':')[2].split(';')))
            X.append(dimensions)
            y.append(taux)

    return np.array(X), np.array(y)


def random_strategy(A):
    """
    """

    score = np.zeros(NB_ANNONCEURS)
    for line in A:
        r = np.random.randint(NB_ANNONCEURS)
        score[r] += line[r]

    return score


def static_best_strategy(A):
    """
    """

    return np.sum(A, axis=0)


def optimal_strategy(A):
    """
    """

    score = np.zeros(NB_ANNONCEURS)
    for line in A:
        r = np.argmax(line)
        score[r] += line[r]

    return score


def argmax_score(score):
    """
    """

    return np.argmax(score) + 1


if __name__ == "__main__":

    A = read_file('CTR.txt')
    random_best = argmax_score(random_strategy(A))
    static_best = argmax_score(static_best_strategy(A))
    optimal_best = argmax_score(optimal_strategy(A))
    print(" Random Strategy : ", random_best)
    print(" StaticBest Strategy : ", static_best)
    print(" Optimal Strategy :", optimal_best)
