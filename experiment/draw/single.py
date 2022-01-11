"""
@author: JUMP
@date 2022/01/03
@description: 单维推荐结果图
"""
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # 解决中文问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = True

    plt.figure(figsize=(18, 5), dpi=150)
    plt.figure(1)

    x_axis = ["F@50", "F@200", "F@400", "F@600"]

    ax1 = plt.subplot(131)
    ax1.tick_params(labelsize=12)

    cora_pr = [0.0383, 0.1220, 0.1766, 0.2239]
    cora_lr = [0.0307, 0.1071, 0.1674, 0.2276]
    cora_his = [0.1877, 0.5863, 0.8853, 0.8806]
    cora_wpr = [0.0345, 0.1101, 0.1812, 0.1996]

    plt.plot(x_axis, cora_pr, color="blue", linestyle="--", label="PR")
    plt.plot(x_axis, cora_lr, color="green", linestyle="-", label="LR")
    plt.plot(x_axis, cora_wpr, color="m", linestyle=":", label="WPR")
    plt.plot(x_axis, cora_his, color="red", linestyle="-.", label="His")

    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.title("Cora [D]", fontsize=18)
    plt.legend()
    plt.xlabel("F@k", fontsize=18)
    plt.ylabel("F-score", fontsize=15)


    ax2 = plt.subplot(132)
    ax2.tick_params(labelsize=12)
    citeseer_pr = [0.0458, 0.1118, 0.1672, 0.1892]
    citeseer_lr = [0.0366, 0.0894, 0.1493, 0.1892]
    citeseer_his = [0.1527, 0.4969, 0.7801, 0.9593]
    citeseer_wpr = [0.0031, 0.0298, 0.0697, 0.1112]

    plt.plot(x_axis, citeseer_pr, color="blue", linestyle="--", label="PR")
    plt.plot(x_axis, citeseer_lr, color="green", linestyle="-", label="LR")
    plt.plot(x_axis, citeseer_wpr, color="m", linestyle=":", label="WPR")
    plt.plot(x_axis, citeseer_his, color="red", linestyle="-.", label="His")

    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.title("Citeseer [D]", fontsize=18)
    plt.legend()
    plt.xlabel("F@k", fontsize=18)
    plt.ylabel("F-score", fontsize=15)


    ax3 = plt.subplot(133)
    ax3.tick_params(labelsize=12)
    pubmed_pr = [0.0020, 0.0137, 0.0351, 0.0531]
    pubmed_lr = [0.0008, 0.0063, 0.0151, 0.0327]
    pubmed_his = [0.0202, 0.0776, 0.1483, 0.2116]
    pubmed_wpr = [0.0024, 0.0204, 0.0408, 0.0618]

    plt.plot(x_axis, pubmed_pr, color="blue", linestyle="--", label="PR")
    plt.plot(x_axis, pubmed_lr, color="green", linestyle="-", label="LR")
    plt.plot(x_axis, pubmed_wpr, color="m", linestyle=":", label="WPR")
    plt.plot(x_axis, pubmed_his, color="red", linestyle="-.", label="His")

    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.title("PubMed [A]", fontsize=18)
    plt.legend()
    plt.xlabel("F@k", fontsize=18)
    plt.ylabel("F-score", fontsize=15)

    # plt.show()
    plt.savefig(fname="single.png", bbox_inches='tight')

