"""
@author: JUMP
@date 2021/12/16
@description: 不带权重的PageRank算法
"""
from experiment.base import Base
import numpy as np


class PageRank(Base):

    def __init__(self, length):
        """init"""
        Base.__init__(self, length)

    def rank(self, adj, epoch):
        """
        排序

        Parameters
        --------
        adj : DiGraph
              有向图的邻接矩阵 横向代表出度，即引用了某篇论文的其他论文

        epoch : int
                迭代轮次


        Return
        ------
        无返回结果，将排序结果存放在self.rank_list中

        """
        d = 0.85
        N = len(adj.nodes())
        q = (1 / float(N))

        for _ in range(epoch):

            tmp_list = [0] * N
            # 记录两次迭代间的差值
            change = 0

            for key in adj.nodes():
                rank_sum = 0

                # 节点key的入度边集合
                in_links = adj.in_edges(key)
                for n in in_links:
                    # 节点的PR值均分给出度节点
                    # 节点的新PR值为其获得的PR值之和
                    outs = len(adj.out_edges(n[0]))
                    rank_sum += (1 / float(outs)) * self.rank_list[n[0]][1]

                tmp = rank_sum * d + (1 - d) * q
                tmp_list[key] = tmp

                change += abs(tmp - self.rank_list[key][1])

            for i in range(N):
                self.rank_list[i][1] = tmp_list[i]

            self.epoch += 1
            self.epoch_record[self.epoch] = change

        print("PageRank", self.epoch, "epoch finished!")




