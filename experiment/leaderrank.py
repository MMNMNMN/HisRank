"""
@author: JUMP
@date 2021/12/16
@description: 不带权重的leaderrank
"""
from experiment.base import Base
import numpy as np


class LeaderRank(Base):

    def __init__(self):
        # LR值字典，key：节点 value：节点的LR值
        self.LR = dict()

        # 已迭代的轮次
        self.epoch = 0

        # key:迭代轮次 value:与前次迭代的差值 用于对比收敛情况
        self.epoch_record = dict()

    def rank(self, adj, epoch):
        """
        排序

        Parameters
        --------
        adj : DiGraph
             邻接矩阵

        epoch : int
                迭代轮次


        Return
        ------
        无返回结果，将排序结果存放在self.rank_list中

        """

        N = len(adj.nodes())
        graph = adj.copy()
        # 增加一个背景节点
        graph.add_node(-1)

        for node in graph.nodes():
            # 背景节点与所有节点双向连接
            graph.add_edge(-1, node)
            graph.add_edge(node, -1)
            # 所有节点LR初始值为1
            self.LR[node] = 1

        # 背景节点初始LR值为0
        self.LR[-1] = 0

        for _ in range(epoch):

            tmp_dict = dict()
            # 记录两次迭代间的差值
            change = 0

            for node in graph.nodes():
                rank_sum = 0
                in_links = graph.in_edges(node)
                # 节点的LR值将均分给其出度节点
                for n in in_links:
                    outs = len(graph.out_edges(n[0]))
                    rank_sum += self.LR[n[0]] / float(outs)

                tmp_dict[node] = rank_sum

            for key in tmp_dict.keys():
                change += abs(self.LR[key] - tmp_dict[key])

            self.LR = tmp_dict

            self.epoch += 1
            self.epoch_record[self.epoch] = change

            if _ % 10 == 0:
                print("LeaderRank", _, "epoch finished!")

        # 迭代结束后，背景节点的LR均分给其余所有节点
        avg = self.LR[-1] / float(N)
        # 删除背景节点
        self.LR.pop(-1)
        for key in self.LR.keys():
            self.LR[key] += avg

    def get_rank(self, predict_label):
        """
        返回排序结果list
        """
        tmp_list = sorted(self.LR.items(), key=lambda x: x[1], reverse=True)
        sorted_list = []
        for item in tmp_list:
            sorted_list.append([predict_label[item[0]], int(item[0]), item[1]])

        return sorted_list

            





