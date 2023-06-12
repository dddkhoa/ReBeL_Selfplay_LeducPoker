import logging

import coloredlogs

from src.rebel.Coach import Coach
from src.stratego_env.Game import Game
from src.rebel.NeuralNet import NeuralNet as nn
from src.rebel.utils import dotdict

from src.stratego_env.stratego_multiagent_env import ObservationModes, GameVersions


log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


args = dotdict({
    'numIters': 10,
    'numEps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 10000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 15,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 20,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    config = {
        'version': GameVersions.STANDARD,
        'random_player_assignment': True,
        'human_inits': True,
        'observation_mode': ObservationModes.PARTIALLY_OBSERVABLE,
    }

    log.info('Loading %s...', Game.__name__)
    g = Game(env_config=config)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
