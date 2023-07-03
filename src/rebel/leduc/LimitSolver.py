import torch
import numpy as np
from typing import List
from copy import copy

from Game import bfs_tree
from config import *

Strategy = List[List[List[float]]]


class CFRParam:
    def __init__(self):
        self.linear = False
        self.rm_plus = False
        self.discount = False
        self.hedge = False
        self.alpha = 1
        self.beta = 1
        self.gamma = 1


class InformationSet:
    def __init__(self, n_act, board, param):
        self.n_act = n_act
        self.board = board
        self.param = param
        self.regret_sum = np.zeros((N_CARD, n_act))
        self.strategy_sum = np.zeros((N_CARD, n_act))
        self.eta = np.sqrt(np.log(n_act)) / 3
        if param.hedge:
            self.ev_mean = np.zeros(N_CARD)
            self.ev_var = np.zeros(N_CARD)
        self.iter_ = 0

    def clear(self):
        self.regret_sum.fill(0)
        self.strategy_sum.fill(0)
        self.iter_ = 0
        if self.param.hedge:
            self.ev_mean.fill(0)
            self.ev_var.fill(0)

    def average_strategy(self):
        return self.norm_strategy(self.strategy_sum)

    def curr_strategy(self):
        if self.param.hedge and self.iter_ >= 2:
            strategy = (self.regret_sum * (self.eta / (np.sqrt(self.iter_) * np.sqrt(self.ev_var)))).exp()
            strategy /= strategy.sum(axis=1)[:, np.newaxis]
            if self.board != N_CARD:
                strategy[self.board] = 1 / self.n_act
            return strategy
        if self.param.rm_plus:
            return self.norm_strategy(self.regret_sum)
        data = self.regret_sum.max(axis=1)
        return self.norm_strategy(data)

    def curr_strategy(self, hand):
        strategy = self.regret_sum[hand]
        if not self.param.rm_plus:
            strategy = strategy.max(0)
        act_sum = strategy.sum()
        if act_sum == 0:
            return np.full(self.n_act, 1 / self.n_act)
        else:
            return strategy / act_sum

    def update_6(self, cfv, child_cfv, strategy, reach_prob):
        if self.param.linear:
            coef = self.iter_ / (self.iter_ + 1)
            self.regret_sum = self.regret_sum * coef + (child_cfv - cfv)
            self.strategy_sum = self.strategy_sum * coef + strategy * reach_prob
        else:
            for i in range(child_cfv.shape[0]):
                self.regret_sum += child_cfv[i][0] - cfv[i]
                self.strategy_sum += strategy[i][0] * reach_prob[0]
        if self.param.rm_plus:
            self.regret_sum = np.maximum(self.regret_sum, 0)
        self.iter_ += 1

    def update(self, regret, weighted_strategy, ev, iter_1):
        if self.param.linear:
            coef = self.iter_ / (self.iter_ + 1)
            self.regret_sum = self.regret_sum * coef + regret
            self.strategy_sum = self.strategy_sum * coef + weighted_strategy
        else:
            self.regret_sum += regret
            self.strategy_sum += weighted_strategy
        if self.param.rm_plus:
            self.regret_sum = np.maximum(self.regret_sum, 0)
        if self.param.hedge:
            self.update_mean_var(ev)
        self.iter_ += 1

    def update_regret(self, hand, regret, iter_):
        if self.param.linear:
            self.regret_sum[hand] *= (self.iter_ - 1) / self.iter_
        self.regret_sum[hand] += regret
        if self.param.rm_plus:
            self.regret_sum[hand] = np.maximum(self.regret_sum[hand], 0)

    def update_avg_strategy(self, hand, weighted_strategy, iter_):
        if self.param.linear:
            self.strategy_sum[hand] *= (self.iter_ - 1) / self.iter_
        self.strategy_sum[hand] += weighted_strategy

    def update_mean_var(self, ev):
        if self.iter_ == 0:
            self.ev_mean += (ev - self.ev_mean) / (self.iter_ + 1)
        else:
            new_mean = self.ev_mean + (ev - self.ev_mean) / (self.iter_ + 1)
            self.ev_var = ((self.iter_ - 1) * self.ev_var + self.iter_ * np.square(self.ev_mean - new_mean) + np.square(
                ev - new_mean)) / self.iter_
            self.ev_mean = new_mean

    def norm_strategy(self, data):
        uniform = 1 / self.n_act
        norm = data.sum(axis=1)
        strategy = data / norm[:, np.newaxis]
        for i in range(N_CARD):
            if norm[i] == 0:
                strategy[i].fill(uniform)
        return strategy


class SolverParam:
    def __init__(self, n_thread=52, max_iter=1000, print_interval=10, accuracy=0.001):
        self.n_thread = n_thread
        self.max_iter = max_iter
        self.print_interval = print_interval
        self.accuracy = accuracy


def other_sum(mat: np.ndarray, reach_prob: np.ndarray, player: int) -> np.ndarray:
    return np.matmul(mat, reach_prob[:, 1 - player])


class LimitSolver:
    def __init__(self, game, cfr_param, param, net=None):
        self.builder = game.builder
        self.hand_mask = game.hand_mask
        self.call_value = game.call_value
        self.cfr_param = cfr_param
        self.net = net if net is not None else None
        self.task = AVG_STRATEGY | BEST_CFV

        if len(self.hand_mask) != N_CARD + 1:
            raise RuntimeError("hand mask matrix size error")
        if len(self.call_value) != N_CARD:
            raise RuntimeError("call value matrix size error")

        self.max_iter_ = param.max_iter
        self.print_interval = param.print_interval
        self.accuracy = param.accuracy
        self.root_cfv_mean = np.zeros((N_CARD, N_PLAYER))
        self.chance_prob = game.chance_prob
        size = len(self.hand_mask)
        self.tensor_mask = []
        n = N_CARD * N_CARD
        for i in range(size):
            self.tensor_mask.append(torch.zeros((N_CARD, N_CARD), dtype=torch.float64))
            self.tensor_mask[i] = copy(torch.tensor(self.hand_mask[i]))

    def set_subtree_data(self, root, init_prob, round_limit=np.inf, depth_limit=np.inf):
        if not root:
            raise RuntimeError("root node is null")

        self.root = root
        self.init_prob = init_prob
        self.norm = 0
        mask = self.hand_mask[root.board]
        for i in range(N_CARD):
            for j in range(N_CARD):
                self.norm += init_prob[i, 0] * init_prob[j, 1] * mask[i, j]

        if self.norm == 0:
            raise RuntimeError("root node is unreachable (reach probability is 0)")

        self.builder.dfs_to_bfs(self.root, bfs_tree, round_limit, depth_limit)
        size = len(bfs_tree)
        self.cfr_data = [None] * size
        self.curr_strategy = [None] * size

        for i in range(size):
            if bfs_tree[i].is_leaf() or bfs_tree[i].node.type != ACTION_NODE:
                continue
            node = bfs_tree[i].node
            n_act = len(node.children)
            temp_prob = 1 / n_act
            self.curr_strategy[i] = np.full((N_CARD, n_act), temp_prob)
            data = InformationSet(n_act, node.board, self.cfr_param)
            self.cfr_data[i] = data

        self.avg_strategy = self.curr_strategy[:]
        self.root_cfv_mean.fill(0)
        self.iter_ = 0

    def step(self):
        cfv_coef = 2 / (self.iter_ + 2) if self.cfr_param.linear else 1 / (self.iter_ + 1)
        for player in range(N_PLAYER):
            root_cfv = self.cfr(player, self.root, self.init_prob, 0)
            self.root_cfv_mean[:, player] += cfv_coef * (root_cfv - self.root_cfv_mean[:, player])
        self.iter_ += 1

    def multi_step(self, iter_s):
        if iter_s == -1:
            iter_s = self.max_iter_
        while self.iter_ < iter_s:
            self.step()

    def train(self):
        e_threshold = (self.root.pots[0] + self.root.pots[1]) * self.accuracy / 100
        e = self.exploitability()
        res = (e[0] + e[1]) / 2
        print("")
        if res < e_threshold:
            return res

        for iter_ in range(self.max_iter_):
            self.step()
            if iter_ % self.print_interval == 0:
                e = self.exploitability()
                res = (e[0] + e[1]) / 2
                print("iter_: %d, exploitability: %f" % (iter_, res))
                if res < e_threshold:
                    return res

        e = self.exploitability()
        res = (e[0] + e[1]) / 2
        return res

    def exploitability(self, outer_strategy):
        if outer_strategy.size != self.avg_strategy.size:
            raise RuntimeError("strategy size error")
        self.avg_strategy, outer_strategy = outer_strategy, self.avg_strategy
        e = self.exploitability(False)
        self.avg_strategy, outer_strategy = outer_strategy, self.avg_strategy
        return e

    def exploitability(self, update=True):
        self.task = AVG_STRATEGY | BEST_CFV
        if update:
            self.get_avg_strategy()
        root_cfv = self.cfv(self.root, self.init_prob, 0)
        value = []
        for player in range(N_PLAYER):
            value.append((root_cfv[:, player] * self.init_prob[:, player]).sum() / self.norm)
        return value

    def get_root_value(self, player, cfv=True):
        if cfv:
            return self.root_cfv_mean[:][player]
        opp_prob = other_sum(self.hand_mask[self.root.board], self.init_prob, player)

        ev = np.zeros(self.root_cfv_mean.shape[0])
        for i in range(self.root_cfv_mean.shape[0]):
            ev[i] = self.root_cfv_mean[i][player] / opp_prob[i]
        for i in range(N_CARD):
            if opp_prob[i] == 0:
                ev[i] = 0
        return ev

    def get_feature(self, node, reach_prob):
        if node.pots[0] != node.pots[1]:
            raise RuntimeError("the bets of both players must be equal")
        act_player = node.player
        query_size = self.net.query_size()

        feature = torch.zeros(query_size, dtype=torch.float32)
        feature[1] = act_player if node.type != CHANCE_NODE else 0
        feature[2] = node.pots[0]  # Both players' accumulated bets

        if node.board != N_CARD:
            feature[3 + node.board] = 1  # One-hot encoding for the community card feature
        prob_sum = np.sum(reach_prob, axis=1)
        prob_feature = torch.from_numpy(reach_prob / prob_sum[:, None])
        dst_ptr = feature + np.full(feature.shape, (3 + N_CARD))

        for player in range(N_PLAYER):
            if prob_sum[player] != 0:
                for i in range(N_CARD):
                    dst_ptr[i] = prob_feature[i][player]
            dst_ptr += N_CARD
            prob_feature += N_CARD

        return feature

    def add_training_data_0(self):
        if self.root.type != ACTION_NODE or self.root.player != 0 or self.root.pots[0] != self.root.pots[1]:
            raise RuntimeError("only support node at the start of betting round")

        query = self.get_feature(self.root, self.init_prob)

        for player in range(N_PLAYER):
            ev = self.get_root_value(player, False)
            node_ev = torch.zeros(N_CARD, dtype=torch.float32)
            node_ev.copy_(torch.tensor(ev, dtype=torch.float32))
            query[0] = player
            self.net.add_training_data(query.clone(), node_ev)

    def add_training_data(self, node, reach_prob):
        if node.type != CHANCE_NODE:
            raise RuntimeError("only support node at the end of betting round")

        next_round_player = 0
        query_size = self.net.query_size()
        query = self.get_feature(node, reach_prob)
        child_feature = torch.zeros(query_size, dtype=torch.float32)
        child_feature[1] = next_round_player
        child_feature[2] = node.pots[0]

        child_query = child_feature.repeat(N_CARD, 1)
        acc = child_query.numpy()
        offset = 3

        for i in range(N_CARD):
            acc[i][offset + i] = 1

        offset += N_CARD
        prob_sum = reach_prob.sum(axis=0)

        for player in range(N_PLAYER):
            sum = prob_sum[player]

            if sum == 0:
                continue

            for i in range(N_CARD):
                prob = reach_prob[:, player].copy()
                temp_sum = sum - prob[i]
                prob[i] = 0

                if temp_sum == 0:
                    continue

                prob /= temp_sum

                for j in range(N_CARD):
                    acc[i][offset + j] = prob[j]

            offset += N_CARD

        opp_prob = torch.zeros(N_CARD, dtype=torch.float64)
        dst_ptr = opp_prob.data_ptr()

        for player in range(N_PLAYER):
            opp_prob.copy_(torch.tensor(reach_prob[:, 1 - player]))
            hand_prob = self.tensor_mask[N_CARD].matmul(opp_prob)
            child_query[:, 0] = player
            query[0] = player
            child_ev = self.net.compute_value(child_query)
            ev = torch.zeros(N_CARD, dtype=torch.float64)

            for i in range(N_CARD):
                ev += self.tensor_mask[i].matmul(opp_prob) * child_ev[i]

            ev *= self.chance_prob
            ev /= hand_prob
            ev.masked_fill_(hand_prob.eq(0), 0)
            self.net.add_training_data(query.clone(), ev.to(torch.float32))

    def get_tree(self):
        return bfs_tree

    def get_avg_strategy(self):
        for i in range(len(bfs_tree)):
            if bfs_tree[i].is_leaf() or bfs_tree[i].node.type != ACTION_NODE:
                continue
            self.avg_strategy[i] = self.cfr_data[i].get_avarage_strategy()
        return self.avg_strategy

    def get_sampling_strategy(self):
        return self.curr_strategy

    def get_belief_propagation_strategy(self):
        return self.curr_strategy

    def set_data(self, node, depth=0, idx=0):
        round = node.round
        type = node.type
        if type == FOLD_NODE or type == SHOWDOWN_NODE:
            return
        n_act = len(node.children)
        if type == ACTION_NODE:
            board = node.board
            data = InformationSet(n_act, board, self.cfr_param)
            self.cfr_data.append(data)
            for i in range(n_act):
                self.set_data(node.children[i], depth + 1)
        elif type == "CHANCE_NODE":
            for i in range(n_act):
                self.set_data(node.children[i], depth + 1)
        else:
            raise RuntimeError("Unknown node type")

    def fold_node_cfv_4(self, player, node, reach_prob):
        fold_p = 1 - node.player
        ev = node.pots[fold_p]
        if player == fold_p:
            ev = -ev
        node_cfv = other_sum(self.hand_mask[node.board], reach_prob, player)
        node_cfv *= ev
        return node_cfv

    def fold_node_cfv(self, node, reach_prob):
        fold_p = 1 - node.player
        ev = node.pots[fold_p]
        node_cfv = np.zeros((N_CARD, N_PLAYER))
        for player in range(N_PLAYER):
            node_cfv[:, player] = other_sum(self.hand_mask[node.board], reach_prob, player)
        node_cfv[:, node.player] *= ev
        node_cfv[:, fold_p] *= -ev
        return node_cfv

    def showdown_node_cfv(self, player, node, reach_prob):
        node_cfv = other_sum(self.call_value[node.board], reach_prob, player)
        node_cfv *= node.pots[0]
        return node_cfv

    def showdown_node_cfv(self, node, reach_prob):
        node_cfv = np.zeros((N_CARD, N_PLAYER))
        for player in range(N_PLAYER):
            node_cfv[:, player] = other_sum(self, self.call_value[node.board], reach_prob, player)
        node_cfv *= node.pots[0]
        return node_cfv

    def cfr(self, player, node, reach_prob, idx=0):
        type = node.type
        if type == FOLD_NODE:
            return self.fold_node_cfv_4(player, node, reach_prob)
        elif type == SHOWDOWN_NODE:
            return self.showdown_node_cfv(player, node, reach_prob)
        elif bfs_tree[idx].is_leaf():
            if self.net is None:
                raise RuntimeError("Need value net")
            if self.net.return_cfv():
                return self.net.compute_value(node, reach_prob, player)
            if node.type != CHANCE_NODE:
                raise RuntimeError("Only query net at chance node")
            query = self.get_feature(node, reach_prob)
            query[0] = player
            node_ev = self.net.compute_value(query.reshape(1, -1)).flatten()
            node_cfv = np.zeros(N_CARD)
            node_cfv[:] = node_ev.numpy()
            node_cfv *= other_sum(self.hand_mask[node.board], reach_prob, player)
        elif type == ACTION_NODE:
            return self.action_node_cfr(player, node, reach_prob, idx)
        elif type == CHANCE_NODE:
            return self.chance_node_cfr(player, node, reach_prob, idx)
        else:
            raise RuntimeError("Unknown node type")

    def cfv(self, node, reach_prob, idx=0):
        type = node.type
        if type == FOLD_NODE:
            return self.fold_node_cfv(node, reach_prob)
        elif type == SHOWDOWN_NODE:
            return self.showdown_node_cfv(node, reach_prob)
        elif bfs_tree[idx].is_leaf():
            if self.net is None:
                raise RuntimeError("Need value net")
            if self.net.return_cfv():
                return self.net.compute_value(node, reach_prob)
            if node.type != CHANCE_NODE:
                raise RuntimeError("Only query net at chance node")
            query = self.get_feature(node, reach_prob).repeat(N_PLAYER, 1)
            for player in range(N_PLAYER):
                query[player][0] = player
            print(query)
            node_ev = self.net.compute_value(query)
            acc = node_ev.accessor()
            node_cfv = np.zeros((N_CARD, N_PLAYER))
            for player in range(N_PLAYER):
                ev = np.zeros(N_CARD)
                for i in range(N_CARD):
                    ev[i] = acc[player][i]
                node_cfv[:, player] = ev * other_sum(self.hand_mask[node.board], reach_prob, player)
            return node_cfv
        elif type == ACTION_NODE:
            return self.action_node_cfv(node, reach_prob, idx)
        elif type == CHANCE_NODE:
            return self.chance_node_cfv(node, reach_prob, idx)
        else:
            raise RuntimeError("Unknown node type")

    def action_node_cfv(self, node, reach_prob, idx):

        player = node.player
        opp = 1 - player
        n_act = len(node.children)
        child_begin = bfs_tree[idx].child_begin
        strategy = self.avg_strategy[idx] if self.task and AVG_STRATEGY else self.curr_strategy[idx]
        my_cfv = np.zeros((N_CARD, n_act))
        opp_cfv = np.zeros((N_CARD, n_act))
        new_prob = reach_prob.copy()
        for i in range(n_act):
            new_prob[:, player] = reach_prob[:, player] * strategy[:, i]
            value = self.cfv(node.children[i], new_prob, child_begin + i)
            my_cfv[:, i] = value[:, player]
            opp_cfv[:, i] = value[:, opp]
        node_cfv = np.zeros((N_CARD, N_PLAYER))
        node_cfv[:, opp] = np.sum(opp_cfv, axis=1)
        if self.task & BEST_CFV:
            node_cfv[:, player] = np.max(my_cfv, axis=1)
        else:
            node_cfv[:, player] = np.sum(my_cfv * strategy, axis=1)
        return node_cfv

    def chance_node_cfv(self, node, reach_prob, idx):
        p0_cfv = np.zeros((N_CARD, N_CARD))
        p1_cfv = np.zeros((N_CARD, N_CARD))
        child_begin = bfs_tree[idx].child_begin
        for i in range(N_CARD):
            new_prob = reach_prob * self.chance_prob
            new_prob[i, :] = 0
            value = self.cfv(node.children[i], new_prob, child_begin + i)
            p0_cfv[:, i] = value[:, 0]
            p1_cfv[:, i] = value[:, 1]
        node_cfv = np.zeros((N_CARD, N_PLAYER))
        node_cfv[:, 0] = np.sum(p0_cfv, axis=1)
        node_cfv[:, 1] = np.sum(p1_cfv, axis=1)
        return node_cfv

    def action_node_cfr(self, player, node, reach_prob, idx):
        node_player = node.player
        n_act = len(node.children)
        child_begin = bfs_tree[idx].child_begin
        data = self.cfr_data[idx]
        strategy = self.curr_strategy[idx]
        new_prob = reach_prob.copy()
        child_cfv = np.zeros((N_CARD, n_act))
        for i in range(n_act):
            new_prob[:, node_player] = reach_prob[:, node_player] * strategy[:, i]
            child_cfv[:, i] = self.cfr(player, node.children[i], new_prob, child_begin + i)
        if player != node_player:
            return np.sum(child_cfv, axis=1)
        if self.cfr_param.hedge:
            node_cfv = np.sum(child_cfv * strategy, axis=1)
            ev = other_sum(self.hand_mask[node.board], reach_prob, player)
            for i in range(N_CARD):
                if ev[i] != 0:
                    ev[i] = node_cfv[i] / ev[i]
            regret = child_cfv - node_cfv[:, np.newaxis]
            weighted_strategy = strategy * reach_prob[:, player]
            data.update(regret, weighted_strategy, ev, data.iter_ + 1)
        else:
            node_cfv = np.sum(child_cfv * strategy, axis=1)
            my_prob = reach_prob[:, player]
            data.update_6(node_cfv, child_cfv, strategy, my_prob)
        # strategy = data.curr_strategy()
        return node_cfv

    def chance_node_cfr(self, player, node, reach_prob, idx):
        opp = 1 - player
        child_begin = bfs_tree[idx].child_begin
        node_cfv = np.zeros((N_CARD, N_CARD))
        for i in range(N_CARD):
            new_prob = reach_prob.copy()
            new_prob[i, :] = 0
            new_prob[:, opp] *= self.chance_prob
            node_cfv[:, i] = self.cfr(player, node.children[i], new_prob, child_begin + i)
        return np.sum(node_cfv, axis=1)
