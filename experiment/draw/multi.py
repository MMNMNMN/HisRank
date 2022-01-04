"""
@author: JUMP
@date 2022/01/03
@description: 多维推荐结果图
"""
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # 解决中文问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = True

    plt.figure(figsize=(12, 4), dpi=150)
    plt.figure(1)

    x_axis = ["F@50", "F@200", "F@400", "F@600"]

    ax1 = plt.subplot(131)
    ax1.tick_params(labelsize=12)

    cora_pr = [0.0361, 0.1289, 0.2285, 0.3049]
    cora_lr = [0.0301, 0.1202, 0.2240, 0.2907]
    cora_his = [0.0922, 0.3293, 0.5712, 0.7506]
    cora_wpr = [0.0281, 0.1307, 0.2285, 0.2791]

    plt.plot(x_axis, cora_pr, color="blue", linestyle="--", label="PR")
    plt.plot(x_axis, cora_lr, color="green", linestyle="-", label="LR")
    plt.plot(x_axis, cora_wpr, color="m", linestyle=":", label="WPR")
    plt.plot(x_axis, cora_his, color="red", linestyle="-.", label="His")
    plt.title("Cora [D,E,F]", fontsize=15)
    plt.legend()


    ax2 = plt.subplot(132)
    ax2.tick_params(labelsize=12)
    citeseer_pr = [0.0269, 0.0917, 0.1921, 0.2592]
    citeseer_lr = [0.0183, 0.0658, 0.1550, 0.2368]
    citeseer_his = [0.0539, 0.1983, 0.3598, 0.4952]
    citeseer_wpr = [0.0258, 0.1036, 0.1903, 0.2651]

    plt.plot(x_axis, citeseer_pr, color="blue", linestyle="--", label="PR")
    plt.plot(x_axis, citeseer_lr, color="green", linestyle="-", label="LR")
    plt.plot(x_axis, citeseer_wpr, color="m", linestyle=":", label="WPR")
    plt.plot(x_axis, citeseer_his, color="red", linestyle="-.", label="His")
    plt.title("Citeseer [C,D,E]", fontsize=15)
    plt.legend()


    ax3 = plt.subplot(133)
    ax3.tick_params(labelsize=12)
    pubmed_pr = [0.0058, 0.0211, 0.0408, 0.0601]
    pubmed_lr = [0.0044, 0.0152, 0.0279, 0.0449]
    pubmed_his = [0.0075, 0.0300, 0.0592, 0.0876]
    pubmed_wpr = [0.0057, 0.0223, 0.0427, 0.0623]

    plt.plot(x_axis, pubmed_pr, color="blue", linestyle="--", label="PR")
    plt.plot(x_axis, pubmed_lr, color="green", linestyle="-", label="LR")
    plt.plot(x_axis, pubmed_wpr, color="m", linestyle=":", label="WPR")
    plt.plot(x_axis, pubmed_his, color="red", linestyle="-.", label="His")
    plt.title("PubMed [A,B]", fontsize=15)
    plt.legend()

    # plt.show()
    plt.savefig(fname="multi.png", bbox_inches='tight')