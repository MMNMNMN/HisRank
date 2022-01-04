"""
@author: JUMP
@date 2021/12/16
@description: 本算法
"""
from experiment.base import Base


class MyRank(Base):

    def __init__(self, user, adj, cos_sim_mat, predict_label, type_num, length):
        """
        init

        Parameters
        ---------
        user : User
                用户模型，记录用户三个时间段的浏览记录

        adj : DiGraph
            邻接矩阵

        cos_sim_mat : ndarray[][]
                    边权重

        predict_label : list
                        各节点类型预测结果，用于计算跳转概率

        type_num : int
                  节点类型数量

        """
        Base.__init__(self, length)

        self.type_num = type_num

        self.user = user

        self.map = self.PR_distribution(adj, cos_sim_mat, predict_label)

        self.p = self.get_probability(predict_label)

    def get_probability(self, predict_label):
        """
        根据浏览记录，获取各类节点的访问概率

        Paramter
        --------
        predict_label : list
                        各节点的预测结果

        Return
        ------
        list
        """
        visit_probability = [0] * self.type_num
        count = [0] * self.type_num

        # 近一周的浏览记录权重为4，近一个月的权重是2，近三个月的权重是1
        sum = 4 * len(self.user.last_week) + 2 * len(self.user.last_month) + len(self.user.last_three_months)

        for i in self.user.last_week:
            count[i] += 4

        for i in self.user.last_month:
            count[i] += 2

        for i in self.user.last_three_months:
            count[i] += 1

        for i in range(self.type_num):
            visit_probability[i] = count[i] / sum

        nums = [0] * self.type_num
        for i in predict_label:
            nums[i] += 1

        p = 0
        for i in range(self.type_num):
            p += visit_probability[i] * (1 / float(nums[i]))

        return p

    def PR_distribution(self, adj, cos_sim_mat, predict_labels):
        """
        计算带权情况下PR值的分配情况

        map : dict
            key:某节点，value：dict
            内层dict里的key代表外层key的出度节点
            内存dict里的value代表外层key把自身PR值以某个比例分给内层key

        """

        map = dict()
        for key in adj.nodes():
            tmp_map = dict()

            # 当前节点的类别
            cur_label = predict_labels[key]
            # 节点的出度边集合
            out_nodes = adj.out_edges(key)

            sum = 0
            # 先求出节点所有出度边的权重总和
            for n in out_nodes:
                out_label = predict_labels[n[1]]

                if cos_sim_mat[cur_label][out_label] == 0:
                    sum += cos_sim_mat[out_label][cur_label]
                else:
                    sum += cos_sim_mat[cur_label][out_label]

            # 求出度节点的PR占比，边权重 / sum
            for n in out_nodes:
                out_label = predict_labels[n[1]]
                if cos_sim_mat[cur_label][out_label] == 0:
                    cur_weight = cos_sim_mat[out_label][cur_label]
                else:
                    cur_weight = cos_sim_mat[cur_label][out_label]

                tmp_map[n[1]] = cur_weight / sum

            map[key] = tmp_map

        return map

    def rank(self, adj, epoch, predict_labels):
        """
        新提出的排序算法

        Parameters
        -------
        adj : DiGraph
              有向图的邻接矩阵 横向代表出度，即引用了某篇论文的其他论文

        predict_labels : ndarray
              预测结果

        epoch : int
              迭代轮次

        Return
        -----
        无返回结果，将排序结果存放在self.rank_list中
        """
        d = 0.85
        map = self.map
        N = len(adj.nodes())

        count = [0] * self.type_num

        for i in self.user.last_week:
            count[i] = 1
        for i in self.user.last_month:
            count[i] = 1
        for i in self.user.last_three_months:
            count[i] = 1

        for _ in range(epoch):

            tmp_list = [0] * N
            # 记录两次迭代间的差值
            change = 0
            for key in adj.nodes():
                rank_sum = 0

                # 节点key的入度边集合
                in_links = adj.in_edges(key)
                for n in in_links:
                    # 节点的PR值按比例分给出度节点
                    rank_sum += map[n[0]][key] * self.rank_list[n[0]][1]

                tmp = rank_sum * d
                if count[predict_labels[key]] == 1:
                    tmp += (1 - d) * float(self.p)

                tmp_list[key] = tmp
                change += abs(tmp - self.rank_list[key][1])

            for i in range(N):
                self.rank_list[i][1] = tmp_list[i]

            self.epoch += 1
            self.epoch_record[self.epoch] = change
        print("MyRank", self.epoch, "epoch finished!")


