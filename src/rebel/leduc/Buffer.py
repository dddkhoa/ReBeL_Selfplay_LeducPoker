from collections import namedtuple, deque
import random

from RecursiveSolver import RecursiveSolver


class ReplayBuffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""
    def __init__(self, buffer_size, batch_size, seed=42):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("TrainSample", field_names=["feature", "target"])
        random.seed(seed)

    def add_experience(self, feature, target):
        """Adds experience(s) into the replay buffer"""
        experience = self.experience(feature, target)
        self.memory.append(experience)

    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def pop_until(self, new_size):
        if new_size >= len(self):
            return
        while len(self) > new_size:
            self.memory.pop()

    def __len__(self):
        return len(self.memory)


class DataLoop:
    def __init__(self, game, cfr_param, param, net, seed=42):
        self.game = game
        self.cfr_param = cfr_param
        self.param = param
        self.net = net
        self.seed = seed
        self.solver = RecursiveSolver(game, cfr_param, param, net)

    def loop(self):
        print("Start loop:")
        self.solver.step()
