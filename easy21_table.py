import numpy as np
from numba import njit


def create_v():
    v = np.zeros((10, 21))
    return v


def create_q():
    q = np.zeros((10, 21, 2))
    return q


@njit
def set_v(v, dealer_score, player_score, value):
    v[dealer_score - 1, player_score - 1] = value
    return v


@njit
def set_q(q, dealer_score, player_score, action, qvalue):
    q[dealer_score - 1, player_score - 1, action] = qvalue
    return q


@njit
def get_v(v, dealer_score, player_score):
    return v[dealer_score - 1, player_score - 1]


@njit
def get_q(q, dealer_score, player_score, action):
    return q[dealer_score - 1, player_score - 1, action]
