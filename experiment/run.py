"""
@author: JUMP
@date 2021/12/13
@description: 加载模型进行预测，使用排序算法对节点进行排序并保存排序结果
"""
import json

import tensorflow as tf
from experiment.exp_utils import *
from gcn.models import GCN
import networkx as nx
from experiment.pagerank import PageRank
from experiment.leaderrank import LeaderRank
from experiment.hits import HITS
from experiment.myrank import MyRank
from experiment.User import User
from experiment.weightedpagerank import WeightedPagerank
import time


class NpEncoder(json.JSONEncoder):
    """
    用户将ndarry保存至json时的数据类型转换
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return  int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


if __name__ == "__main__":

    dataset = "cora"
    type_num = 7

    # dataset = "citeseer"
    # type_num = 6

    # dataset = "pubmed"
    # type_num = 3



    flags = tf.app.flags
    # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('dataset', dataset, 'Dataset string.')
    # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_string('model', 'gcn', 'Model string.')
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

    adj, features, y_test, test_mask = load_data(dataset)

    # 未处理过的原始特征矩阵，已转换为ndarray
    raw_features = features.A
    # 未处理过的原始邻接矩阵，已转换为DiGraph
    raw_adj = nx.DiGraph(adj)


    # Some preprocessing
    features = preprocess_features(features)
    support = [preprocess_adj(adj)]

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(1)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_test.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = GCN(placeholders, input_dim=features[2][1], logging=True, name=dataset)

    # Initialize session
    sess = tf.Session()
    # Init variables
    sess.run(tf.global_variables_initializer())

    # 加载模型
    print("Loading model start...")
    saverdir = "../model/" + dataset
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(saverdir))
    print("Loading model finish...")

    # 节点类型预测
    print("Model predict start...")
    feed_dict_validate = construct_feed_dict(features, support, y_test, test_mask, placeholders)
    predictions = sess.run(model.predict(), feed_dict=feed_dict_validate)
    print("Model predict finish...")

    # 将预测结果转换成标签0 ~ n
    predict_labels = transfer_res_to_label(predictions)
    # 获取各类的代表向量
    repre_vectors = get_representation_vector(raw_features, predict_labels,type_num)
    # 获取代表向量间的相似度矩阵
    cos_sim_mat = get_cos_sim_matrix(repre_vectors, type_num)



    """
    PageRank算法
    cora:209轮达到稳态
    citeseer:220轮左右
    pubmed:204轮
    """

    # total = 0
    # for _ in range(5):
    #     pr_start = time.time()
    #     pageRank = PageRank(adj.shape[0])
    #     for i in range(20):
    #         pageRank.rank(raw_adj, 10)
    #     pr_list = pageRank.get_rank(predict_labels)
    #     pr_end = time.time()
    #     total += pr_end - pr_start
    # print("pagerank avg time:", total / 5)


    # pr_record = pageRank.get_epoch_record()
    # pr_path = "./" + dataset + "_rank/pr_rank.json"
    # with open(pr_path, "w") as f:
    #     json.dump(pr_list, f, cls=NpEncoder)
    # pr_end = time.time()
    # print("pagerank total time:", pr_end - pr_start)


    #

    """
    本算法
    single:
    cora: 218轮
    last_week = [4]
    
    citeseer:206轮
    last_week = [4]
    
    pubmed:217轮
    last_week = [0]
    
    
    multi:
    cora: 4、5、6 三类 235轮
            last_week = [4, 5, 6]
            last_month = [6,6,6,5,5]
            last_three_months = [4,4]
    
    
    citeseer:3、4、5 三类 230轮
            last_week = [3, 4, 5]
            last_month = [5,5,5,4,4]
            last_three_months = [3,3]
    
    pubmed:0、1 两类 233轮
            last_week = [0,1,0]
            last_month = []
            last_three_months = []
            
            
    personalized recommendation:
    cora: 234轮   personal_1.json
            last_week = [4, 6]
            last_month = [6,6,6,4,4]
            last_three_months = [6,6]
            
            233轮  personal_2.json
            last_week = [4, 4]
            last_month = [4, 4, 4, 4, 6]
            last_three_months = [4, 4]
            
    """
    last_week = [0]
    last_month = []
    last_three_months = []

    # myrank_total_time = 0
    # for _ in range(5):
    #     myrank_start = time.time()
    #     user = User(last_week, last_month, last_three_months)
    #     myrank = MyRank(user, raw_adj, cos_sim_mat, predict_labels, type_num, adj.shape[0])
    #     for i in range(20):
    #         myrank.rank(raw_adj, 10, predict_labels)
    #     myrank_list = myrank.get_rank(predict_labels)
    #     myrank_end = time.time()
    #     myrank_total_time += (myrank_end - myrank_start)
    # print("MyRank average runtime:", myrank_total_time / 5)

    # myrank_record = myrank.get_epoch_record()
    # myrank_path = "./" + dataset + "_rank/personal_2.json"
    # with open(myrank_path, "w") as f:
    #     json.dump(myrank_list, f, cls=NpEncoder)




    """
    HITS算法
    cora:100轮
    citeseer:100轮
    pubmed:100轮
    """

    # total = 0
    # for _ in range(2):
    #     hits_start = time.time()
    #     hits = HITS(raw_adj, adj.shape[0])
    #     for i in range(20):
    #         hits.rank(raw_adj, 10)
    #     hist_list = hits.get_rank(predict_labels)
    #     hits_end = time.time()
    #     total += hits_end - hits_start
    # print("hits runtime:", total / 2)

    # hits_record = hits.get_epoch_record()
    # hits_path = "./" + dataset + "_rank/hits_rank.json"
    # with open(hits_path, "w") as f:
    #     json.dump(hist_list, f, cls=NpEncoder)


    """
    LeaderRank算法
    cora:167轮
    citeseer:180轮
    pubmed:268轮
    """

    # total = 0
    # for _ in range(5):
    #     lr_start = time.time()
    #     leader_rank = LeaderRank()
    #     leader_rank.rank(raw_adj, 200)
    #     lr_list = leader_rank.get_rank(predict_labels)
    #     lr_end = time.time()
    #     total += lr_end - lr_start
    # print("lr runtime:", total / 5)

    # lr_record = leader_rank.get_epoch_record()
    # lr_path = "./" + dataset + "_rank/lr_rank.json"
    # with open(lr_path, "w") as f:
    #     json.dump(lr_list, f, cls=NpEncoder)


    """
    Weighted PageRank
    cora:100轮
    citeseer:100轮
    pubmed:50轮
    """
    total = 0
    for _ in range(5):
        wpr_start = time.time()
        wpr = WeightedPagerank(raw_adj, adj.shape[0])
        for i in range(10):
            wpr.rank(raw_adj, 10)
        wpr_list = wpr.get_rank(predict_labels)
        wpr_end = time.time()
        total += wpr_end - wpr_start
    print("wpr runtime:", total / 5)


    # wpr_records = wpr.get_epoch_record()
    # wpr_path = "./" + dataset + "_rank/wpr_rank.json"
    # with open(wpr_path, "w") as f:
    #     json.dump(wpr_list, f, cls=NpEncoder)

    i = 1


