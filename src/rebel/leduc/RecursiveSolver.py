import numpy as np
import random
from collections import deque

from LimitSolver import LimitSolver
# from Game import bfs_tree
from config import *


class RecursiveSolver:
    def __init__(self, game, cfr_param, param, net):
        self.game = game
        self.max_iter = param.max_iter
        self.solver = LimitSolver(game, cfr_param, param, net)
        self.curr = game.root
        self.epsilon = 0.25
        self.sample_path = []

    def sample_from_root(self):
        mask = self.game.hand_mask[self.curr.board]
        num = N_CARD * N_CARD
        weight = np.zeros(shape=num)
        for i in range(N_CARD):
            for j in range(N_CARD):
                weight[i * N_CARD + j] = self.init_prob[i][0] * self.init_prob[j][1] * mask[i][j]
        return np.random.choice(range(num), p=weight/np.sum(weight))

    def sample_leaf(self):
        self.bfs_tree = self.solver.get_tree()
        sampling_strategy = self.solver.get_sampling_strategy()
        propogation_strategy = self.solver.get_belief_propagation_strategy()
        idx = 0
        player = np.random.randint(0, 2)
        distribution = np.random.default_rng().uniform(0, 1)
        sample_belief = self.init_prob.copy()
        h = self.sample_from_root()
        hand = [h // N_CARD, h % N_CARD]
        is_chance = self.curr.type == CHANCE_NODE
        if is_chance:
            self.sample_path.append([self.curr, self.init_prob])
        while (not self.bfs_tree[idx].is_leaf()) or is_chance:
            c = np.random.uniform(0, 1)
            act = 0
            n_act = len(self.curr.children)
            node_p = self.curr.player
            if is_chance:
                weight = np.ones(N_CARD)
                weight[hand[0]] = weight[hand[1]] = 0
                act = np.random.choice(np.arange(N_CARD), p=weight / np.sum(weight))
            elif (node_p == player and c < self.epsilon):
                act = np.random.randint(0, n_act)
            else:
                act_prob = sampling_strategy[idx][hand[node_p]].copy()
                act = np.random.choice(np.arange(n_act), p=act_prob)
            if is_chance:
                sample_belief[act, :] = 0
                self.init_prob[act, :] = 0
            else:
                sample_belief[:, node_p] *= sampling_strategy[idx][:, act]
                self.init_prob[:, node_p] *= propogation_strategy[idx][:, act]
            idx = self.bfs_tree[idx].child_begin + act
            self.curr = self.curr.children[act]
            pre_is_chance = is_chance
            is_chance = self.curr.type == CHANCE_NODE
            if is_chance:
                self.sample_path.append((self.curr, self.init_prob))
            if pre_is_chance:
                break

    def step(self):
        self.init_prob = self.game.init_prob
        iter = 0
        self.curr = self.game.root
        skip_sample = False
        while not self.curr.is_leaf():
            prob_sum = np.sum(self.init_prob, axis=0)
            if prob_sum[0] == 0 or prob_sum[1] == 0:
                break
            self.init_prob /= prob_sum
            self.solver.set_subtree_data(self.curr, self.init_prob, 1)
            sample_iter = 0
            while iter < sample_iter:
                self.solver.step()
                iter += 1
            self.sample_path.clear()
            skip_sample = self.curr.round == N_ROUND-1
            if not skip_sample:
                self.sample_leaf()
            while iter < self.max_iter:
                self.solver.step()
                iter += 1
            self.solver.add_training_data_0()
            for node, prob in self.sample_path:
                self.solver.add_training_data(node, prob)
            if skip_sample:
                break


class FullStrategySolver:
    def __init__(self, game, cfr_param, param, net, sampling_iter, use_sampling_strategy):
        self.game = game
        self.solver = LimitSolver(game, cfr_param, param, net)
        self.max_iter = param.max_iter
        self.sampling_iter = sampling_iter
        self.use_sampling_strategy = use_sampling_strategy

        game.builder.dfs_to_bfs(game.root, self.bfs_tree)
        self.reset(1)

        if self.sampling_iter:
            weight = [i + 1 for i in range(self.max_iter)]
            self.generator = random.choices(range(self.max_iter), weights=weight, k=self.max_iter)

    def reset(self, seed):
        random.seed(seed)
        size = len(self.bfs_tree)
        self.strategy = [None] * size

        for i in range(size):
            if self.bfs_tree[i].is_leaf() or self.bfs_tree[i].node.type != ACTION_NODE:
                continue

            n_act = len(self.bfs_tree[i].node.children)
            self.strategy[i] = np.full((N_CARD, n_act), 1 / n_act)

    def get_full_strategy(self):
        self.recursive_solving(self.game.init_prob, 0)
        return self.strategy

    def recursive_solving(self, reach_prob, idx=0):
        self.solver.set_subtree_data(self.bfs_tree[idx].node, reach_prob, 1)
        sample_iter = self.generator[idx] + 1 if self.sampling_iter else self.max_iter
        self.solver.multi_step(sample_iter)

        part_strategy = self.solver.get_sampling_strategy() if self.use_sampling_strategy else self.solver.get_avg_strategy()
        belief_strategy = self.solver.get_belief_propogation_strategy() if self.use_sampling_strategy else self.solver.get_avg_strategy()
        part_bfs_tree = self.solver.get_tree()
        size = len(part_bfs_tree)

        dq = deque([(idx, 0, reach_prob)])
        leaf_dq = deque()

        while dq:
            full_id, part_id, prob = dq.popleft()
            bfs_node = self.bfs_tree[full_id]

            if bfs_node.node != part_bfs_tree[part_id].node:
                raise RuntimeError("Subtree error")

            if part_bfs_tree[part_id].is_leaf():
                if not bfs_node.node.is_leaf():
                    leaf_dq.append((full_id, part_id, prob))
                continue

            self.strategy[full_id] = part_strategy[part_id]
            child_begin = bfs_node.child_begin
            part_child_begin = part_bfs_tree[part_id].child_begin
            n_act = bfs_node.child_end - child_begin
            player = bfs_node.node.player
            is_chance = bfs_node.node.type == CHANCE_NODE

            for i in range(n_act):
                new_prob = prob.copy()

                if is_chance:
                    new_prob[i, :] = 0
                else:
                    new_prob[:, player] *= part_strategy[part_id][:, i]

                dq.append((child_begin + i, part_child_begin + i, new_prob))

        for full_id, _, prob in leaf_dq:
            prob_sum = np.sum(prob, axis=0)

            if prob_sum[0] == 0 or prob_sum[1] == 0:
                continue

            prob /= prob_sum
            self.recursive_solving(prob, full_id)
