import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from stellargraph import StellarGraph
import networkx as nx
from networkx.readwrite import json_graph
import pandas as pd
import json
import os
import seaborn as sns


AML_ALL_TYPE_DICT = {None: 0, 'path': 1, 'star': 2, 'ring': 3, 'clique': 4, 'tree': 5}
AML_ANOMALY_FLAG_TYPE_DICT = {None: 0, 'path': 1, 'star': 1, 'ring': 1, 'clique': 1, 'tree': 1}

categories_features = {
    "Localise": set(['comb_abs90', 'comb_absolute_value', 'comb_exp_1', 'comb_exp_2', 
                     'comb_exp_3', 'comb_exp_4', 'comb_ipr90', 'comb_ipr_1', 'comb_ipr_2', 
                     'comb_ipr_3', 'comb_ipr_4', 'comb_sign_equal_1', 'comb_sign_equal_2', 
                     'comb_sign_stat_1', 'comb_sign_stat_2', 'lower_abs90', 'lower_absolute_value', 
                     'lower_exp_1', 'lower_exp_2', 'lower_exp_3', 'lower_exp_4', 'lower_ipr90', 
                     'lower_ipr_1', 'lower_ipr_2', 'lower_ipr_3', 'lower_ipr_4', 'lower_sign_equal_1', 
                     'lower_sign_equal_2', 'lower_sign_stat_1', 'lower_sign_stat_2', 'rw_abs90', 
                     'rw_absolute_value', 'rw_exp_1', 'rw_exp_2', 'rw_exp_3', 'rw_exp_4', 'rw_ipr90', 
                     'rw_ipr_1', 'rw_ipr_2', 'rw_ipr_3', 'rw_ipr_4', 'rw_sign_equal_1', 'rw_sign_equal_2', 
                     'rw_sign_stat_1', 'rw_sign_stat_2', 'upper_abs90', 'upper_absolute_value', 
                     'upper_exp_1', 'upper_exp_2', 'upper_exp_3', 'upper_exp_4', 'upper_ipr90', 
                     'upper_ipr_1', 'upper_ipr_2', 'upper_ipr_3', 'upper_ipr_4', 'upper_sign_equal_1', 
                     'upper_sign_equal_2', 'upper_sign_stat_1', 'upper_sign_stat_2']),
    "NetEMD": set(['NetEMD_comb_1', 'NetEMD_comb_2', 'NetEMD_lower_1', 'NetEMD_lower_2', 
                   'NetEMD_rw_1', 'NetEMD_rw_2', 'NetEMD_upper_1', 'NetEMD_upper_2', 
                   'in_out_strength_1', 'in_out_strength_2', 'in_strength_1', 'in_strength_2',
                   'out_strength_1', 'out_strength_2', 'motif_10_1', 'motif_10_2', 'motif_11_1', 
                   'motif_11_2', 'motif_12_1', 'motif_12_2', 'motif_13_1', 'motif_13_2', 
                   'motif_14_1', 'motif_14_2', 'motif_15_1', 'motif_15_2', 'motif_16_1', 
                   'motif_16_2', 'motif_4_1', 'motif_4_2', 'motif_5_1', 'motif_5_2', 'motif_6_1', 
                   'motif_6_2', 'motif_7_1', 'motif_7_2', 'motif_8_1', 'motif_8_2', 'motif_9_1', 
                   'motif_9_2']),    
    "Path": set(['path_10', 'path_11', 'path_12', 'path_13', 'path_14', 'path_15', 'path_16', 
                 'path_17', 'path_18', 'path_19', 'path_2', 'path_20', 'path_3', 'path_4', 
                 'path_5', 'path_6', 'path_7', 'path_8', 'path_9']),
    "Com. Density": set(['degree_std', 'first_density', 'first_strength', 'second_density', 
                         'second_strength', 'small_community', 'third_density']),
    "Basic": set(['gaw10_score', 'gaw20_score', 'gaw_score'])
}



def concatenate_XY(Xs, Ys):
    """Concatenates DataFrame or Series.
    """
    features = Xs[0].columns
    X = pd.DataFrame(columns=features)
    Y = pd.Series(name='type')
    for X_, Y_ in zip(Xs, Ys):
        X = pd.concat([X, X_], axis=0, ignore_index=True, sort=True)
        Y = Y.append(Y_, ignore_index=True)
    return X, Y


def split_train_test(data, test_size=0.3, mix_all=True, select_features=None):
    """Splits data into training set and testing set.
    If mix_all is True, all node features (rows) of different
    networks are mixed and splited. Otherwise, some networks
    are kept entirely for testing purpose.
    """

    Ys = [X['type'] for X in data]
    Xs = [X.drop('type', axis=1) for X in data]
    if select_features is not None:
        Xs = [X.loc[:, select_features] for X in Xs]
    if mix_all:
        X, Y = concatenate_XY(Xs, Ys)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    else:
        # keep complete graph for test
        num_test_graph = max(0, int(len(data)*test_size))
        test_indice = np.random.choice(range(len(data)), size=num_test_graph, replace=False)
        Xs_train = [X for i, X in enumerate(Xs) if i not in test_indice]
        Ys_train = [Y for i, Y in enumerate(Ys) if i not in test_indice]
        X_train, Y_train = concatenate_XY(Xs_train, Ys_train)
        X_test = [X for i, X in enumerate(Xs) if i in test_indice]
        Y_test = [Y for i, Y in enumerate(Ys) if i in test_indice]
    return X_train, X_test, Y_train, Y_test


def precision_recall(preds, labels, *sample_sizes):
    """Computes the precision, recall and F1-score for a prediciton, for different sample sizes.
    """
    sorted_label_pred = sorted(zip(labels, preds), key=lambda x: x[1], reverse=True)
    sorted_labels = np.array([l for l, p in sorted_label_pred])
    num_anomalies = np.sum(labels)
    precs = []
    recs = []
    f1s = []
    for sample_size in sample_sizes:
        num_anormaly_samples = np.sum(sorted_labels[:sample_size])
        prec = num_anormaly_samples / sample_size
        rec = num_anormaly_samples / num_anomalies
        f1 = 0 if prec+rec==0 else 2 * prec * rec / (prec + rec)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    if len(recs) == 1:
        return precs[0], recs[0], f1s[0]
    else:
        return precs, recs, f1s


def plot_metrics(history):
  metrics =  ['loss', 'auc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()


def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

  plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
  plt.xlabel('False positives [%]')
  plt.ylabel('True positives [%]')
  plt.xlim([-0.5,100.5])
  plt.ylim([-0.5,100.5])
  plt.grid(True)
  ax = plt.gca()
  ax.set_aspect('equal') 
   

def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))

    
def average_precision(preds, labels):
    sample_sizes = list(range(1, len(labels)+1))
    precs, recs, _ = precision_recall(preds, labels, *sample_sizes)
    avg_p = 0
    for i in range(len(precs)-1):
        p, r = precs[i], recs[i]
        r_next = recs[i+1]
        avg_p += p * (r_next - r)
    return avg_p


def feature_sum_orig(X, normalize=False):
    """Sum features of each node of X.
    If normalize is True, scaling before summing.
    => this original implementation has issue with zero division
    => see below re-write feature_sum() using normalize from sklearn having no issue with zero division
    """
    if normalize:
        X_ = X.copy()
        X_min = X_.min(axis=0)
        X_max = X_.max(axis=0)
        X_scale = X_max - X_min
        X_ = np.where(X_scale != 0, (X_ - X_min) / X_scale, 0.0)
        return X_.sum(axis=1)
    return X.sum(axis=1)


def feature_sum(X, normalize=False):
    """
    => this impl. replace the above orginal impl. to overcome division by zero issue
    Sum features of each node of X.
    If normalize is True, scaling before summing.
    """
    if normalize:
        X_ = X.copy()
        X_ = preprocessing.normalize(X_)
        return X_.sum(axis=1)
    return X.sum(axis=1)


def generate_node_id_map(id_offset, graph_node_id_list):
    return {i: i + id_offset for i in graph_node_id_list}


def load_networkx_graphs(data_path):
    i = 0
    max_node_id = -1
    combined_G = nx.DiGraph()
    for data_file in os.listdir(data_path):
        print(data_file, ' - max_node_id=', max_node_id)
        data_file_path = os.path.join(data_path, data_file)
        G = json_graph.node_link_graph(json.load(open(data_file_path)), directed=True)
        
        node_id_map = generate_node_id_map(max_node_id + 1, list(G.nodes))
        G = nx.relabel_nodes(G, node_id_map)
        combined_G = nx.compose(combined_G, G)
        max_node_id = max(combined_G.nodes)
        i = i + 1
    return combined_G


def retrieve_node_features_with_labels(data_file_path):
    G = json_graph.node_link_graph(json.load(open(data_file_path)), directed=True)
    g_df = pd.DataFrame.from_dict(G.nodes, orient='index')
    g_df = g_df.fillna(0)    
    return g_df


def retrieve_node_features_and_labels_tuple(networkx_graph):
    g_df = pd.DataFrame.from_dict(networkx_graph.nodes, orient='index')
    g_df = g_df.fillna(0)
    anomaly_types = g_df['type']
    labels = (g_df['type'] != 0).astype(int)
    g_df = g_df.drop('type', axis=1)
    g_df = g_df.reindex(sorted(g_df.columns), axis=1)
    return g_df, labels, anomaly_types


def generate_stellar_graph_old(combined_directed_G):
    node_feature_df, node_type_labels, _ = retrieve_node_features_and_labels_tuple(combined_directed_G)

    networkx_G = nx.DiGraph()
    networkx_G.add_edges_from(combined_directed_G.edges(data=True))

    G = StellarGraph.from_networkx(networkx_G, node_features=node_feature_df,
                                   node_type_default="account", edge_type_default="transaction")
    return G, node_type_labels, node_feature_df


def generate_stellar_graph(combined_directed_G):
    node_feature_df, node_type_labels, _ = retrieve_node_features_and_labels_tuple(combined_directed_G)
    graph_feature = {i: [edge[2]['weight'] for edge in list(combined_directed_G.out_edges(i, data=True))] for i in list(combined_directed_G.nodes)}
    # max_features = max([len(node_feature) for node_feature in graph_feature.values()])    
    max_features = 150 # use fix feature numbers to runable and consistant on every graphs
    for key, value in graph_feature.items():
        if len(value) > max_features:
            graph_feature[key] = value[0: max_features]
        else:
            graph_feature[key].extend([0] * (max_features - len(value)))
    
    networkx_G = nx.DiGraph()
    networkx_G.add_edges_from(combined_directed_G.edges(data=True))
    nx.set_node_attributes(
        networkx_G,
        graph_feature,
        "feature",
    )

    G = StellarGraph.from_networkx(networkx_G, node_features="feature",
                                   node_type_default="account", edge_type_default="transaction")
    return G, node_type_labels, node_feature_df

