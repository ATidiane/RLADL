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


def UCB(dataset):
    """FIXME! briefly describe function

    :param dataset:
    :returns:
    :rtype:

    """
    n_samples = dataset.shape[0]
    mu = np.zeros((dataset.shape[1]))
    # par défaut on initialise à 1 pour ne pas diviser par 0
    count = np.ones((dataset.shape[1]))
    cum = np.zeros((dataset.shape[1]))  # pour stocker les sumcum pour mu
    pi = np.zeros((dataset.shape[0]), int)  # la politiqueUCB

    for t in range(1, dataset.shape[0] + 1):  # l'itération à laquelle on est
        Bt = mu + np.sqrt(2 * np.log(t) / count)
        pi[t - 1] = int(Bt.argmax())
        cum[pi[t - 1]] += dataset[t - 1][pi[t - 1]]
        count[pi[t - 1]] += 1
        mu[pi[t - 1]] = cum[pi[t - 1]] / count[pi[t - 1]]

    return pi


def LinUCB(context, dataset, alpha=0.5):
    """FIXME! briefly describe function

    :param context:
    :param dataset:
    :param alpha:
    :returns:
    :rtype:

    """
    caract_doc = context.shape[1]
    actions = np.zeros((context.shape[0]), int)
    a = np.eye(caract_doc).reshape(
        1, caract_doc, caract_doc).repeat(
        dataset.shape[1], axis=0)
    b = np.zeros((dataset.shape[1], caract_doc))
    theta = np.zeros((dataset.shape[1], caract_doc))
    p = np.zeros((dataset.shape[1]))
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            inva = np.linalg.inv(a[j])
            theta[j] = inva.dot(b[j])
            p[j] = theta[j].dot(context[i]) + alpha * \
                np.sqrt(context[i].dot(inva.dot(context[i])))
        actions[i] = np.random.choice(np.where(p == p.max())[0])
        a[actions[i]] += context[i].dot(context[i])
        b[actions[i]] += dataset[i][actions[i]] * context[i]
    return actions


if __name__ == "__main__":

    A = read_file('CTR.txt')
    random_best = argmax_score(random_strategy(A))
    static_best = argmax_score(static_best_strategy(A))
    optimal_best = argmax_score(optimal_strategy(A))
    print(" Random Strategy : ", random_best)
    print(" StaticBest Strategy : ", static_best)
    print(" Optimal Strategy :", optimal_best)
