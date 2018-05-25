import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def calc_cross_entropy(y, h):
    return -(np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)))


def calc_reg_penalty(w, penalty):
    if penalty == "l2":
        return np.dot(w.transpose(), w)
    else:
        raise Exception("Unknown penalty: {}".format(penalty))


def get_one_hots(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


def get_one_hot(target, nb_classes):
    return get_one_hots(target, nb_classes)[0]
