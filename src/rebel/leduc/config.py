from typing import List

N_PLAYER = 2
N_CARD = 6
N_SUIT = 2
N_RANK = 3
N_ROUND = 2

ACTION_NODE = 0
CHANCE_NODE = 1
FOLD_NODE = 2
SHOWDOWN_NODE = 3

AVG_STRATEGY = 1
BEST_CFV = 2

Actions = ['call', 'raise', 'fold', 'check']
# Strategy = List[List[List[float]]]
