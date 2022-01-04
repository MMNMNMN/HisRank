"""
@author: JUMP
@date 2021/12/16
@description: 基类， 主要是一些对比实验的测试方法
"""
import numpy as np


class Base:

    def __init__(self, length):
        """init"""

        # 已迭代的轮次
        self.epoch = 0

        # key:迭代轮次 value:与前次迭代的差值 用于对比收敛情况
        self.epoch_record = dict()

        # 存放算法排序的结果
        # 二维ndarray, shape : node_num * 2，第一个值代表数据集中的下标，第二个值代表排序得分
        self.rank_list = []

        all_list = []
        init_pr = 1 / float(length)
        for _ in range(length):
            all_list.append([_, init_pr])
        self.rank_list = np.array(all_list, dtype=np.float64)

    def get_rank(self, predict_label):
        """
        获取排序结果
        """
        self.rank_list = self.rank_list[self.rank_list[:, 1].argsort()]
        self.rank_list = self.rank_list[::-1]

        sorted_list = []
        for item in self.rank_list:
            sorted_list.append([predict_label[int(item[0])], int(item[0]), item[1]])
        return sorted_list

    def get_epoch_record(self):
        """
        获取收敛情况
        """
        pre_record = self.epoch_record[1]
        ans = []
        for key, val in self.epoch_record.items():
            ans.append(val / pre_record)
            if val == 0:
                break
            pre_record = val
        return ans

    def test(self):
        """某度量方法"""
        i = 1