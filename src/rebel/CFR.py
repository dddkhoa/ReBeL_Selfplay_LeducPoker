import logging
import numpy as np


log = logging.getLogger(__name__)


class CFR:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.infoSet = ""
        self.canonBoard = None
        self.regretSum = np.zeros(game.getActionSize())
        self.strategy = np.zeros(game.getActionSize())
        self.strategySum = np.zeros(game.getActionSize())

    def get_strategy(self, realizationWeight, canonicalBoard, player):
        actions = self.game.getValidMoves(canonicalBoard, player)
        num_actions = len(actions)
        normalizingSum = 0
        for a in actions:
            self.strategy[a] = self.regretSum[a] if self.regretSum[a] > 0 else 0
            normalizingSum += self.strategy[a]

        for a in actions:
            if normalizingSum > 0:
                self.strategy[a] /= normalizingSum
            else:
                self.strategy[a] = 1.0 / num_actions
            self.strategySum[a] += realizationWeight * self.strategy[a]
        return self.strategy

    # Get average information set mixed strategy across all training iterations i
    def get_average_strategy(self):
        avgStrategy = np.zeros(self.game.getActionSize())
        normalizingSum = 0
        for a in range(self.game.getActionSize()):
            normalizingSum += self.strategySum[a]
        for a in range(self.game.getActionSize()):
            if (normalizingSum > 0):
                avgStrategy[a] = round(self.strategySum[a] / normalizingSum, 2)
            else:
                avgStrategy[a] = round(1.0 / self.game.getActionSize())
        return avgStrategy

    def __str__(self):
        return self.infoSet + ": " + str(self.get_average_strategy())  # + "; regret = " + str(self.regretSum)


