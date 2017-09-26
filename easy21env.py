import random
from numba import njit


possible_dealer_card = ((1, "black"), (2, "black"), (3, "black"), (4, "black"),
                        (5, "black"), (6, "black"), (7, "black"), (8, "black"),
                        (9, "black"), (10, "black"))

possible_player_scores = range(-9, 31)
possible_player_scores_non_terminal = range(1, 21)

color_dict = {"black": 0, "red": 1}
action_dict = {"stick": 0, "take": 1}


@njit
def take(color=-1):
    number = random.randint(1, 10)
    if color == -1:
        color = 1 if random.random() > 0.75 else 0
    return number, color


@njit
def step(dealer_card, player_score, action):
    reward = 0
    # Player take card.
    if action == 1:
        card = take()
        if card[1] == 0:
            player_score += card[0]
        else:
            player_score -= card[0]
        if player_score > 21 or player_score < 1:
            terminal = True
            reward = -1
        else:
            terminal = False
    # Dealer takes all cards until terminal
    else:
        terminal = False
        dealer_score = dealer_card[0]
        while not terminal:
            if dealer_score < 17:
                card = take()
                if card[1] == 0:
                    dealer_score += card[0]
                else:
                    dealer_score -= card[0]
                if dealer_score > 21 or dealer_score < 1:
                    reward = +1
                    terminal = True
                else:
                    terminal = False
            else:
                if player_score == dealer_score:
                    reward = 0
                if player_score > dealer_score:
                    reward = 1
                if player_score < dealer_score:
                    reward = -1
                terminal = True
    # state = {}
    # state["dealer_card"] = dealer_card
    # state["player_score"] = player_score
    # state["terminal"] = terminal
    # state["reward"] = reward
    return dealer_card, player_score, terminal, reward
    # return state


def test_step():
    # take
    for i in range(10000):
        dealer_card = take(0)
        player_score = random.choice(possible_player_scores_non_terminal)
        dealer_card, player_score, terminal, reward = step(dealer_card,
                                                           player_score, 1)
        assert(dealer_card == dealer_card)
        assert(player_score in possible_player_scores)
        if player_score <= 0 or player_score > 21:
            assert(terminal)
            assert(reward == -1)
        else:
            assert(not terminal)
            assert(reward == 0)
    # stick
    for i in range(10000):
        dealer_card = take(0)
        player_score = random.choice(possible_player_scores_non_terminal)
        dealer_card, player_score, terminal, reward = step(dealer_card,
                                                           player_score, 0)
        assert(dealer_card == dealer_card)
        assert(player_score == player_score)
        assert(terminal)
        assert(reward == 1
               or reward == 0
               or reward == -1)


test_step()
