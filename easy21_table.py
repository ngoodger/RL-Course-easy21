import numpy as np
from numba import njit


# Easy 21 Value function table
def create_v():
    v = np.zeros((10, 21))
    return v


# Easy 21 Q Value function table
def create_q():
    q = np.zeros((10, 21, 2))
    return q


# Easy 21 Eligibility trace table
@njit
def create_e():
    e = np.zeros((10, 21, 2))
    return e


@njit
def set_v(v, dealer_score, player_score, value):
    v[dealer_score - 1, player_score - 1] = value
    return v

@njit
def set_q(q, dealer_score, player_score, action, qvalue):
    q[dealer_score - 1, player_score - 1, action] = qvalue
    return q


@njit
def increment_e(e, dealer_score, player_score, action):
    e[dealer_score - 1, player_score - 1, action] += 1
    return e


@njit
def get_v(v, dealer_score, player_score):
    return v[dealer_score - 1, player_score - 1]


@njit
def get_q(q, dealer_score, player_score, action):
    return q[dealer_score - 1, player_score - 1, action]


@njit
def get_e(e, dealer_score, player_score, action):
    return e[dealer_score - 1, player_score - 1, action]


@njit
def decay_e(e, discount_factor, sarsa_lambda):
    return e * discount_factor * sarsa_lambda


@njit
def decay_state_e(e, dealer_score, player_score, action, discount_factor,
                  sarsa_lambda):
    e[dealer_score - 1, player_score - 1, action] *= (discount_factor *
                                                      sarsa_lambda)
    return e


@njit
def reset_e(e):
    return e * 0.0
