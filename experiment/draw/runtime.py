"""
@author: JUMP
@date 2021/12/29
@description: 运行时间复式柱状图
"""
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # 解决中文问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = True

    x = ['PageRank', 'LeaderRank', 'Weighted PR', 'HisRank']

    y_cora = [14.9683, 17.3600, 14.5038, 9.5141]
    y_citeseer = [14.4022, 18.0024, 13.8956, 9.3824]
    y_pubmed = [125.1919, 141.7203, 124.1079, 79.9217]


    width_bar = 0.2

    plt.figure(dpi=100)


    x_1 = list(range(len(x)))
    # 把各种条形图间隔开，防止重叠，都加上width_bar的倍数
    x_2 = [i + width_bar for i in x_1]
    x_3 = [i + width_bar * 2 for i in x_1]

    plt.xticks(x_2, x)

    plt.bar(x_1, y_cora, width=width_bar, label="Cora")
    plt.bar(x_2, y_citeseer, width=width_bar, label="Citeseer")
    plt.bar(x_3, y_pubmed, width=width_bar, label="Pubmed")

    plt.xlabel("算法", fontsize=12)
    plt.ylabel("运行时间 单位：秒", fontsize=12)

    plt.legend()
    plt.savefig(fname="runtime.png", bbox_inches='tight')
