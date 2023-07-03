from typing import Any, Dict, Optional

from LimitSolver import LimitSolver
from config import *

import torch


class ValueNet:
    def compute_value(self, node: Any, reach_prob, player):
        return torch.zeros(N_CARD)

    def compute_value(self, node: Any, reach_prob: torch.Tensor) -> torch.Tensor:
        return torch.zeros(N_PLAYER, N_CARD, dtype=torch.float32)

    def compute_value(self, node: Any, reach_prob: Dict[str, Any]) -> torch.Tensor:
        return torch.zeros(N_CARD, N_PLAYER, dtype=torch.float32)

    def compute_value(self, node: Optional[Any], reach_prob: torch.Tensor, player: int) -> torch.Tensor:
        return torch.zeros(N_CARD, dtype=torch.float32)

    def compute_value(self, query: torch.Tensor) -> torch.Tensor:
        return torch.zeros(query.size(0), N_CARD, dtype=torch.float32)

    def return_cfv(self):
        return False

    def add_training_data(self, query, value):
        pass

    def query_size(self):
        return 1 + 1 + 1 + N_CARD + N_PLAYER * N_CARD


class OracleNet(ValueNet):
    def __init__(self, game, cfr_param, param):
        self.solver = LimitSolver(game, cfr_param, param)
        self.any_zero = False

    def compute_value(self, node, reach_prob, player=None):
        self.solve(node, reach_prob)
        cfv = torch.zeros((N_CARD, N_PLAYER))
        if self.any_zero:
            return cfv
        if player is not None:
            return self.solver.get_root_value(player)
        for play in range(N_PLAYER):
            cfv[:, play] = self.solver.get_root_value(play)
        return cfv

    def return_cfv(self):
        return True

    def solve(self, node, reach_prob):
        prob_sum = reach_prob.sum(axis=0)
        if prob_sum[0] == 0 or prob_sum[1] == 0:
            self.any_zero = True
            return
        self.any_zero = False
        self.solver.set_subtree_data(node, reach_prob)
        self.solver.multi_step(-1)


class TorchScriptNet(ValueNet):
    def __init__(self, path, device):
        self.device = torch.device(device)
        try:
            self.model = torch.jit.load(path)
        except torch.jit.Error as e:
            print(e)
        print("load:", path)
        self.model.to(self.device)

    def compute_value(self, query):
        with torch.no_grad():
            return self.model.forward([query.to(self.device)]).to('cpu')


class TrainDataNet(ValueNet):
    def __init__(self, buffer):
        self.buffer = buffer

    def add_training_data(self, query, value):
        self.buffer.add_experience(query, value)
