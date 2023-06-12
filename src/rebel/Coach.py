import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm


from src.rebel.CFR import CFR
log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.nodeMap = {}
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    # Counterfactual regret minimization iteration
    def cfr(self, board, history, p0, p1, curPlayer):
        # Return payoff for terminal states
        r = self.game.getGameEnded(board, curPlayer)
        if r != 0:
            return r
        # identify the infoset (or node)
        canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
        infoSet = self.game.stringRepresentation(canonicalBoard)

        # Get the node or create it, if it does not exist
        node_exists = infoSet in self.nodeMap
        node = None
        if not node_exists:
            node = CFR(self.game, self.nnet, self.args)
            node.infoSet = infoSet
            node.canonBoard = canonicalBoard
            self.nodeMap[infoSet] = node
        else:
            node = self.nodeMap[infoSet]

        # For each action, recursively call cfr with additional history and probability
        param = p0 if curPlayer == 1 else p1
        strategy = node.get_strategy(param, board, curPlayer)
        actions = self.game.getValidMoves(canonicalBoard, curPlayer)
        print(actions)
        util = self.game.getInitBoard()

        nodeUtil = 0

        for a in actions:
            new_board = self.game.getNextState(board, curPlayer, a)
            nextHistory = history + str(curPlayer)
            # the sign of the util received is the opposite of the one computed one layer below
            # because what is positive for one player, is neagtive for the other
            # if player == 0 is making the call, the reach probability of the node below depends on the strategy of player 0
            # so we pass reach probability = p0 * strategy[a], likewise if this is player == 1 then reach probability = p1 * strategy[a]
            util[a] = - self.cfr(new_board, nextHistory, p0 * strategy[a], p1, 0 - curPlayer) if curPlayer == 1 else - self.cfr(new_board, nextHistory, p0, p1 * strategy[a],  0 - curPlayer)
            nodeUtil += strategy[a] * util[a]

        # For each action, compute and accumulate counterfactual regret
        for a in actions:
            regret = util[a] - nodeUtil
            # for the regret of player 0 is multilied by the reach p1 of player 1
            # because it is the action of player 1 at the layer above that made the current node reachable
            # conversly if this player 1, then the reach p0 is used.
            node.regretSum[a] += (p1 if curPlayer == 1 else p0) * regret

        self.trainExamples.append([node.canonBoard, curPlayer, node.get_average_strategy(), r if r != 0 else None])
        return nodeUtil

    def executeEpisode(self, board):
        """
        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        self.trainExamples = []
        self.cfr(board, "", 1, 1, 1)
        return [(x[0], x[2], x[3]) for x in self.trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    board = self.game.getInitBoard()
                    iterationTrainExamples += self.executeEpisode(board)

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # shuffle examples before training // only for mcts
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            # shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            # nmcts = MCTS(self.game, self.nnet, self.args)

            #if pitting
            """
            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            """
            #if no pitting
            log.info('ACCEPTING NEW MODEL')
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')


    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True