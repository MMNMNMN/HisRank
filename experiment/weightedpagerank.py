"""
@author: JUMP
@date 2021/12/16
@description: 带权重的PageRank
"""
from experiment.base import Base


class WeightedPagerank(Base):
    def __init__(self, adj, length):
        """init"""
        Base.__init__(self, length)
        self.in_map, self.out_map = self.get_map(adj)

    def get_map(self, adj):
        """
        获得各条边的入度权重

        Parameter
        --------
        adj:DiGraph
            邻接矩阵
        """
        in_map = dict()
        out_map = dict()

        for key in adj.nodes():
            tmp_in_map = dict()
            tmp_out_map = dict()

            # 当前节点的出度边集合
            out_nodes = adj.out_edges(key)

            # 当前节点的所有出度节点的入度之和
            in_sum = 0
            # 当前节点的所有出度节点的出度之和
            out_sum = 0
            for n in out_nodes:
                in_sum += len(adj.in_edges(n[1]))
                out_sum += len(adj.out_edges(n[1]))

            for n in out_nodes:
                # 出度节点的入度 / sum
                tmp_in_map[n[1]] = len(adj.in_edges(n[1])) / in_sum
                tmp_out_map[n[1]] = len(adj.out_edges(n[1])) / out_sum

            in_map[key] = tmp_in_map
            out_map[key] = tmp_out_map

        return in_map, out_map

    def rank(self, adj, epoch):
        """
        排序

        Parameters
        --------
        epoch : int
                迭代轮次

        Return
        ------
        无返回结果，将排序结果存放在self.rank_list中

        """

        d = 0.85
        N = len(adj.nodes())

        for _ in range(epoch):

            tmp_list = [0] * N
            # 记录两次迭代间的差值
            change = 0

            for key in adj.nodes():
                rank_sum = 0

                # 节点key的入度边集合
                in_links = adj.in_edges(key)
                for n in in_links:
                    rank_sum += self.rank_list[n[0]][1] * self.out_map[n[0]][key] * self.in_map[n[0]][key]

                tmp = rank_sum + (1 - d)
                tmp_list[key] = tmp

                change += abs(tmp - self.rank_list[key][1])

            for i in range(N):
                self.rank_list[i][1] = tmp_list[i]

            self.epoch += 1
            self.epoch_record[self.epoch] = change

        print("Weighted PageRank", self.epoch, "epoch finished!")


