"""
@author: JUMP
@date 2021/12/13
@description: 实验目录下的工具方法，主要有算法处理模块、对比实验模块和模型加载模块
"""
import sys
import numpy as np
import scipy.sparse as sp
import pickle as pkl
import networkx as nx
import json

# --------------------------------算法处理相关---------------------------------
def transfer_res_to_label(predictions):
    """

    Parameters
    ----------
    predictions : 预测所得结果
                如cora的预测结果是2708 * 7的标签矩阵
                每行代表一个节点，每列代表预测属于该标签的概率，取七个标签中最大概率为预测结果

    Return
    ------
    返回一个一维列表
    一个1 * 2708的预测结果
    """
    ans = []
    for res in predictions:
        ans.append(np.argmax(res))

    return ans


def get_representation_vector(X, predict_labels,type_num):
    """
    对于预测后属于同类的节点，将原始特征矩阵(未处理)内的对应向量累加
    最后标准化作为该类节点的代表向量

    Example
    -------
    累加后的详列列表vector_list = [array([1., 1., 1., 0., 0.]),
                   array([1., 0., 2., 1., 1.])]
    求出的标志向量vector_repres = [arrary([0.3333,  0.3333, 0.33333, 0., 0.]),
                                arrary([0.2, 0., 0.4, 0.2, 0.2])]


    Parameters
    ----------
    X : ndarray
        未标准化的特征矩阵

    predict_labels : list
                     1 * 2708的列表，每行代表各节点的标签预测结果

    type_num : int
                数据集中节点类型数量

    Return
    ------
    repre_vector : list
                    各类节点的代表向量
                    以cora为例，含7个元素，每个元素是1 * 1433的ndarray，分别是七个类别的代表向量
    """

    vetor_list = []
    # 创建一个全零初始数组，数组长度为特征数量
    empty_vector = np.array([i - i for i in range(X.shape[1])])
    # 总共分为type_num类
    for i in range(type_num):
        vetor_list.append(empty_vector)


    # 节点总数是特征矩阵的行数，按行下标遍历
    for index in range(X.shape[0]):
        # 先通过预测结果获取类别的下标
        vector_index = predict_labels[index]
        # 通过类别下标，找到该类节点的向量和，并继续累加
        vetor_list[vector_index] = np.sum([vetor_list[vector_index], X[index]], axis=0)

    repre_vectors = []
    for item in vetor_list:
        repre_vectors.append(item / np.sum(item, axis=0))

    return repre_vectors


def get_cos_sim_matrix(repre_vectors, type_num):
    """
    计算标志向量间的余弦相似度


    Parameters
    ---------
    repre_vectors : list
                    由get_representation_vector()获取

    Return
    ------
    type_num : int
                当前数据集中节点类型数量，如cora有7种类型节点

    cos_sim_mat : numpy.ndarray
                7*7的一个上三角矩阵，[i,j]代表类i和类j的相似度
    """

    # 创建一个对角阵
    cos_sim_mat = np.eye(type_num)

    # 计算向量间的余弦相似度
    for i in range(type_num):
        for j in range(i+1, type_num):
            vector_a = np.mat(repre_vectors[i])
            vector_b = np.mat(repre_vectors[j])
            num = float(vector_a * vector_b.T)
            denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
            cos_sim_mat[i][j] = num / denom

    return cos_sim_mat


# --------------------------------对比实验相关---------------------------------

def compare(dataset, user_interest, algorithm_name, epoch, single):
    """
    和其他排序算法的排序结果进行比较

    Parameters
    ---------
    dataset : str
            数据集名字

    user_interest : list[int]
                  含有0、1的列表，列表长度为数据集中节点类别数
                  为1表示用户访问过某类节点

    algorithm_name : str
                    待比较的算法名字

    epoch : int
            选取PageRank排序结果的前epoch条作为对比参照

    single : bool
            多推荐还是单推荐测试标志位
    """

    pr_path = "./" + dataset + "_rank/" + algorithm_name + "_rank.json"
    with open(pr_path, "r") as f:
        pr_list = json.load(f)

    if single:
        my_path = "./" + dataset + "_rank/myrank_single.json"
    else:
        my_path = "./" + dataset + "_rank/myrank_multi.json"

    with open(my_path, "r") as f:
        my_list = json.load(f)



    count = dict()

    # 记录待比较算法的前epoch条记录中推荐项的排名
    for i in range(epoch):
        if user_interest[pr_list[i][0]] == 1:
            count[pr_list[i][1]] = i

    # 待比较算法中推荐项的排名和同一推荐项在myrank中的排名差
    sum = 0
    for key in count.keys():
        for j in range(len(my_list)):
            if key == my_list[j][1]:
                count[key] = j - count[key]
                sum += count[key]

    print(count)

    new_add = 0
    # 前epoch条记录中,待比较算法没有，而myrank中有的推荐项
    for i in range(epoch):
        if user_interest[my_list[i][0]] == 1 and count.get(my_list[i][1]) is None:
            new_add += 1

    name_map = dict()
    name_map["lr"] = "LeaderRank"
    name_map["pr"] = "PageRank"
    name_map["hits"] = "HITS"
    print(dataset, "下", name_map[algorithm_name], "前", epoch, "条 排名总差额：", sum, "新增推荐项数", new_add)


def compare_personal(dataset, interest_a, interest_b, algorithm_name, epoch):
    """
    测试根据浏览记录个性化推荐效果

    Example
    ------
    两个浏览记录都只浏览了a、b两类节点
    一个浏览记录访问a类的概率是0.9，访问b类节点的概率是0.1，另一个相反，访问a类节点概率0.1，访问b类概率0.9
    根据两个浏览记录生成不同的排序结果，和pr比较，查看排序效果

    """

    base_path = "./" + dataset + "_rank/" + algorithm_name + "_rank.json"
    with open(base_path, "r") as f:
        base_list = json.load(f)

    personal_1_path = "./" + dataset + "_rank/personal_1.json"
    with open(personal_1_path, "r") as f:
        personal_1_list = json.load(f)

    personal_2_path = "./" + dataset + "_rank/personal_2.json"
    with open(personal_2_path, "r") as f:
        personal_2_list = json.load(f)


    # 统计base_ist中两种浏览类别下，各节点的排名
    count = dict()
    for i in range(epoch):
        if base_list[i][0] == interest_a:
            count[base_list[i][1]] = i
        elif base_list[i][0] == interest_b:
            count[base_list[i][1]] = i

    personal_1_a = 0
    personal_1_b = 0

    # personal_1的排序结果与base对比
    for key in count.keys():
        for j in range(len(personal_1_list)):
            if key == personal_1_list[j][1]:
                if personal_1_list[j][0] == interest_a:
                    personal_1_a += j - count[key]
                elif personal_1_list[j][0] == interest_b:
                    personal_1_b += j - count[key]

    print(epoch, "轮 Personal_1 a类节点排名差额：", personal_1_a, "b类节点排名差额：", personal_1_b)


    personal_2_a = 0
    personal_2_b = 0
    for key in count.keys():
        for j in range(len(personal_2_list)):
            if key == personal_2_list[j][1]:
                if personal_2_list[j][0] == interest_a:
                    personal_2_a += j - count[key]
                elif personal_2_list[j][0] == interest_b:
                    personal_2_b += j - count[key]

    print(epoch, "轮 Personal_2 a类节点排名差额：", personal_2_a, "b类节点排名差额：", personal_2_b)


def calc_f1(dataset, user_interest, algo_file_name):
    """
    测试推荐结果的F1值

    Parameter
    --------
    dataset:int
            数据集名

    user_interest: list
                用户感兴趣的节点类型

    algo_file_name: str
                    算法推荐结果文件名

    Return
    ------
    输出 f1值 [0,1] 越大越好
    """
    path = "./" + dataset + "_rank/" + algo_file_name + ".json"

    with open(path, "r") as f:
        result_list = json.load(f)

    # 统计样本总数
    total_sample_num = 0
    for res in result_list:
        total_sample_num += user_interest[res[0]]

    # 命中样本总数
    sample_num = 0

    # 前50条记录
    for i in range(0, 50):
        sample_num += user_interest[result_list[i][0]]

    precision = sample_num / 50
    if total_sample_num != 0:
        recall = sample_num / total_sample_num
    else:
        recall = 0

    if (precision + recall) != 0:
        f1 = (2 * precision * recall)/(precision + recall)
    else:
        f1 = 0

    print(dataset, algo_file_name, "Precision@50:", precision, "Recall@50:", recall,
          "f1@50:", f1)

    # 前200条记录
    for i in range(50, 200):
        sample_num += user_interest[result_list[i][0]]

    precision = sample_num / 200
    if total_sample_num != 0:
        recall = sample_num / total_sample_num
    else:
        recall = 0

    if (precision + recall) != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    print(dataset, algo_file_name, "Precision@200:", precision, "Recall@200:", recall,
          "f1@200:", f1)

    # 前400条记录
    for i in range(200, 400):
        sample_num += user_interest[result_list[i][0]]

    precision = sample_num / 400
    if total_sample_num != 0:
        recall = sample_num / total_sample_num
    else:
        recall = 0

    if (precision + recall) != 0:
        f1 = (2 * precision * recall)/(precision + recall)
    else:
        f1 = 0
    print(dataset, algo_file_name, "Precision@400:", precision, "Recall@400:", recall,
          "f1@400:", f1)

    # 前600条记录
    for i in range(400, 600):
        sample_num += user_interest[result_list[i][0]]

    precision = sample_num / 600
    if total_sample_num != 0:
        recall = sample_num / total_sample_num
    else:
        recall = 0

    if (precision + recall) != 0:
        f1 = (2 * precision * recall)/(precision + recall)
    else:
        f1 = 0
    print(dataset, algo_file_name, "Precision@600:", precision, "Recall@600:", recall,
          "f1@600:", f1)


# --------------------------------加载模型相关---------------------------------
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    加载数据集，用于预测
    基于gcn/utils.py修改，修改加载目录，去除不需要的变量
    """

    print("Loading", dataset_str, "start...")

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("../gcn/data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("../gcn/data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()

    test_mask = sample_mask(idx_test, labels.shape[0])

    y_test = np.zeros(labels.shape)
    y_test[test_mask, :] = labels[test_mask, :]

    print("Loading", dataset_str, "finished...")

    return adj, features, y_test, test_mask


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict



