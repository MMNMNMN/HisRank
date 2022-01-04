"""
@author: JUMP
@date 2021/12/16
@description: HITS算法
"""
from experiment.base import Base
import math


class HITS(Base):

    def __init__(self, adj, length):
        """
        init

        Parameters
        ---------
        adj : DiGraph
              邻接矩阵

        """
        # 已迭代的轮次
        self.epoch = 0

        # key:迭代轮次 value:与前次迭代的差值 用于对比收敛情况
        self.epoch_record = dict()

        self.hub = dict()
        self.authority = dict()
        for node in adj.nodes():
            self.hub[node] = 1
            self.authority[node] = 1

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
        无返回结果，将排序结果存放在self.authority中

        """

        N = len(adj.nodes())

        for _ in range(epoch):
            # 用于auth标准化
            auth_norm = 0
            # 用于hub标准化
            hub_norm = 0

            tmp_hub = dict()
            tmp_auth = dict()

            change = 0

            for node in adj.nodes():

                auth_sum = 0
                hub_sum = 0

                outs = adj.out_edges(node)
                for out in outs:
                    # 节点i的hub值为所有i指向的页面的auth值之和
                    hub_sum += self.authority[out[1]]

                in_links = adj.in_edges(node)
                for n in in_links:
                    # 节点i的auth值为所有指向i的节点的hub之和
                    auth_sum += self.hub[n[0]]

                auth_norm += pow(auth_sum, 2)
                hub_norm += pow(hub_sum, 2)

                # 收敛情况判断
                tmp_hub[node] = hub_sum
                change += abs(hub_sum - self.hub[node])
                tmp_auth[node] = auth_sum
                change += abs(auth_sum - self.authority[node])

            for i in range(N):
                self.hub[i] = tmp_hub[i]
                self.authority[i] = tmp_auth[i]

            # auth标准化
            auth_norm = auth_norm ** 0.5
            for key in self.authority.keys():
                self.authority[key] /= auth_norm

            # hub标准化
            hub_norm = hub_norm ** 0.5
            for key in self.hub.keys():
                self.hub[key] /= hub_norm

            self.epoch += 1
            self.epoch_record[self.epoch] = change

        print("HITS", self.epoch, "epoch finished!")

    def get_rank(self, predict_label):
        """
        返回排序结果list
        """
        tmp_list = sorted(self.authority.items(), key=lambda x : x[1], reverse=True)
        sorted_list = []
        for item in tmp_list:
            sorted_list.append([predict_label[item[0]], int(item[0]), item[1]])
        return sorted_list

