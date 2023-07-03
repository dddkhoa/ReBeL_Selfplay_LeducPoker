import numpy as np
from typing import List
from collections import deque

from config import *


def card2rank(x):
    return x >> 1


def card2suit(x):
    return x & 1


class Node:
    def __init__(self, player, round, p0_pot, p1_pot, type_, board=N_CARD):
        self.player = player
        self.round = round
        self.board = board
        self.pots = [p0_pot, p1_pot]
        self.type = type_
        self.children: List[Node] = []  # child nodes of action node

    def is_leaf(self):
        return self.type == FOLD_NODE or self.type == SHOWDOWN_NODE


class BFSNode:
    def __init__(self, node, depth):
        self.node = node
        self.depth = depth
        self.child_begin = 0
        self.child_end = 0

    def is_leaf(self):
        return self.child_begin == self.child_end


# Global variable for LimitSolver
bfs_tree: List[BFSNode] = []


class Rule:
    def __init__(self):
        self.stop_at_chance = False
        self.init_pot = [1, 1]
        self.raise_ = [2, 4]
        self.max_bet_num = 3

    def get_stack(self):
        stack = self.init_pot[0]
        for i in range(N_ROUND):
            stack += self.raise_[i] * self.max_bet_num
        return stack


class TreeBuilder:
    def __init__(self, rule):
        self.rule = rule

    def build_tree(self, node, bet_num=0):
        if node is None:
            return
        type_ = node.type
        round_ = node.round
        if type_ == FOLD_NODE or type_ == SHOWDOWN_NODE:
            return
        pots = node.pots
        player = node.player
        opp = 1 - player
        my_pot = pots[player]
        opp_pot = pots[opp]
        if my_pot > opp_pot:
            raise RuntimeError("tree error")
        child = None
        if type_ == CHANCE_NODE:
            if self.rule.stop_at_chance:
                return
            if pots[0] != pots[1] or round_ != N_ROUND - 1:
                raise RuntimeError("tree error")
            for i in range(N_CARD):
                child = Node(0, round_, pots[0], pots[1], ACTION_NODE, i)
                node.children.append(child)
                self.build_tree(child, 0)
            return
        if opp_pot > my_pot:  # fold
            child = Node(opp, round_, pots[0], pots[1], FOLD_NODE, node.board)
            node.children.append(child)
        if player == 0 and bet_num == 0:  # initial call
            if my_pot != opp_pot:
                raise RuntimeError("tree error")
            child = Node(opp, round_, opp_pot, opp_pot, ACTION_NODE, node.board)
        elif round_ != N_ROUND - 1:  # not the last round
            child = Node(opp, round_ + 1, opp_pot, opp_pot, CHANCE_NODE, node.board)
        else:
            child = Node(opp, round_, opp_pot, opp_pot, SHOWDOWN_NODE, node.board)
        self.build_tree(child, bet_num)
        node.children.append(child)
        # raise
        if bet_num >= self.rule.max_bet_num:
            return
        p0_pot = opp_pot + self.rule.raise_[round_]
        p1_pot = opp_pot
        if player != 0:
            p0_pot, p1_pot = p1_pot, p0_pot
        child = Node(opp, round_, p0_pot, p1_pot, ACTION_NODE, node.board)
        self.build_tree(child, bet_num + 1)
        node.children.append(child)

    # round_limit, depth_limit increasing
    def dfs_to_bfs(self, node, bfs_tree, round_limit=float('inf'), depth_limit=float('inf')):
        bfs_tree.clear()
        dq = deque([node])
        bfs_tree.append(BFSNode(node, 0))
        idx = 0
        init_round = node.round

        while dq:
            node = dq.popleft()
            if node != bfs_tree[idx].node:
                raise RuntimeError("BFS tree error")

            depth = bfs_tree[idx].depth
            diff = node.round - init_round
            bfs_tree[idx].child_begin = bfs_tree[idx].child_end = len(bfs_tree)

            if node.is_leaf() or diff >= round_limit or depth >= depth_limit:
                idx += 1
                continue

            bfs_tree[idx].child_end += len(node.children)
            for child in node.children:
                dq.append(child)
                bfs_tree.append(BFSNode(child, depth + 1))

            idx += 1


class Game:
    def __init__(self, p0_pot=1, p1_pot=1, raise0=2, raise1=4, max_bet_num=3):
        self.rule = Rule()
        self.builder = TreeBuilder(self.rule)
        self.rule.init_pot[0] = p0_pot
        self.rule.init_pot[1] = p1_pot
        self.rule.raise_[0] = raise0
        self.rule.raise_[1] = raise1
        self.rule.max_bet_num = max_bet_num
        self.root = Node(0, 0, p0_pot, p1_pot, ACTION_NODE)
        self.builder.build_tree(self.root)
        self.chance_prob = 1 / (N_CARD - N_PLAYER)

        self.init_prob = np.full((N_CARD, N_PLAYER), 1 / N_CARD)
        self.hand_mask = [None] * (N_CARD + 1)
        self.hand_mask[N_CARD] = np.ones((N_CARD, N_CARD))
        base_mask = self.hand_mask[N_CARD]
        np.fill_diagonal(base_mask, 0)
        self.call_value = [None] * N_CARD
        for i in range(N_CARD):
            self.hand_mask[i] = np.copy(base_mask)
            self.hand_mask[i][i, :] = 0
            self.hand_mask[i][:, i] = 0
            self.call_value[i] = self.call_value_matrix(i)

    def call_value_matrix(self, board):
        board_rank = card2rank(board)
        hand_rank = [card2rank(i) for i in range(N_CARD)]
        value = np.zeros((N_CARD, N_CARD))
        for i in range(N_CARD):
            if i == board:
                continue
            for j in range(N_CARD):
                if j == i or j == board:
                    continue
                if hand_rank[i] == board_rank:
                    value[i, j] = 1
                elif hand_rank[j] == board_rank:
                    value[i, j] = -1
                elif hand_rank[i] > hand_rank[j]:
                    value[i, j] = 1
                elif hand_rank[i] < hand_rank[j]:
                    value[i, j] = -1
        return value

    def reset(self, rule):
        self.rule.init_pot[0] = rule.init_pot[0]
        self.rule.init_pot[1] = rule.init_pot[1]
        self.rule.raise_[0] = rule.raise_[0]
        self.rule.raise_[1] = rule.raise_[1]
        self.rule.max_bet_num = rule.max_bet_num
        self.root = Node(0, 0, self.rule.init_pot[0], self.rule.init_pot[1], ACTION_NODE)
        self.builder.build_tree(self.root)
