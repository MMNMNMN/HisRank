"""
@author: JUMP
@date 2021/12/19
@description: 
"""
from experiment.exp_utils import calc_f1


if __name__ == "__main__":


    # algorithm_file_name = "myrank_multi"
    algorithm_file_name = "wpr_rank"

    # 用户浏览记录中访问过的类型统计
    # user_interest = [0,0,0,0,1,1,1]
    # user_interest = [0, 0, 0, 0, 1, 0, 0, 0]
    # dataset = "cora"
    #
    # dataset = "citeseer"
    # user_interest = [0,0,0,1,1,1]
    # user_interest = [0, 0, 0, 0, 1, 0, 0]

    dataset = "pubmed"
    user_interest = [1,1,0]
    # user_interest = [1, 0, 0]

    calc_f1(dataset, user_interest, algorithm_file_name)


