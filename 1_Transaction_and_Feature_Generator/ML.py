import matplotlib
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
# %matplotlib inline

from rpy2.robjects.packages import importr
devtools = importr('devtools')
# devtools.install_github("dynverse/netdist", dependencies = True)

from utils import generate_null_models, get_parameters
from generator import ER_generator, draw_anomalies
from basic_test import basic_features
from com_detection import community_detection
from spectral_localisation import spectral_features
from NetEMD import NetEMD_features
from path_finder import path_features

import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from utils import precision_recall, average_precision
from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn import preprocessing


def concatenate_XY(Xs, Ys):
    """Concatenates DataFrame or Series.
    """
    features = Xs[0].columns
    X = pd.DataFrame(columns=features)
    Y = pd.Series(name='type')
    for X_, Y_ in zip(Xs, Ys):
        # concat each dataset (each CSV file) in the "list" Xs to a long dataset X
        X = pd.concat([X, X_], axis=0, ignore_index=True, sort=True)
        # append the labels of each dataset (each CSV file) to a long pandas series labels
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
        if test_size:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
        else:
            X_train, X_test, Y_train, Y_test = (X, [], Y, [])
    else:
        # keep complete graph for test
        num_test_graph = max(1, int(len(data)*test_size))
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


def plot_precision_recall_f1(sample_sizes, precs, recs, f1):
    # plot
    plt.figure(figsize=(9, 4))

    plt.subplot(131)
    plt.plot(sample_sizes, precs)
    plt.xscale('log')
    plt.title("precision")
    plt.xlabel('No. nodes')

    plt.subplot(132)
    plt.plot(sample_sizes, recs)
    plt.xscale('log')
    plt.title("recall")
    plt.xlabel('No. nodes')

    plt.subplot(133)
    plt.plot(sample_sizes, f1)
    plt.xscale('log')
    plt.title("F1-score")
    plt.xlabel('No. nodes')


####### loading data ########

# list all data files
data_path = './data/'
os.listdir(data_path)

data_path = './data/'
data = []
for data_file in os.listdir(data_path):
    data_file_path = os.path.join(data_path, data_file)
    X = pd.read_csv(data_file_path, index_col=0)
    X['type'] = (X['type']!='0').astype(int)
    data.append(X)


######## Feature Sum ########

# load dataset
X, _, Y, _ = split_train_test(data, test_size=None)

# Feature sum without normalization
pred_fs = feature_sum(X, normalize=False)
sample_sizes = [8, 16, 32, 64, 128, 256, 512, 1014]
precs, recs, f1 = precision_recall(pred_fs, Y, *sample_sizes)

print("Total number of samples: {}\n".format(len(Y)))
print("{}\t{}\t{}\t{}".format('samples', 'precision', 'recall', 'f1_score'))
print('-'*40)
for samples, prec, rec, f_1 in zip(sample_sizes, precs, recs, f1):
    print('{}\t{:.3f}\t\t{:.3f}\t{:.3f}'.format(samples, prec, rec, f_1))

avg_prec = average_precision(pred_fs, Y)
avg_prec

# plot result of feature sum without normalization
plot_precision_recall_f1(sample_sizes, precs, recs, f1)


######## Feature sum with normalization ########
pred_fs_norm = feature_sum(X, normalize=True)
sample_sizes = [8, 16, 32, 64, 128, 256]
precs, recs, f1 = precision_recall(pred_fs_norm, Y, *sample_sizes)

print("Total number of samples: {}\n".format(len(Y)))
print("{}\t{}\t{}\t{}".format('samples', 'precision', 'recall', 'f1_score'))
print('-'*40)
for samples, prec, rec, f_1 in zip(sample_sizes, precs, recs, f1):
    print('{}\t{:.3f}\t\t{:.3f}\t{:.3f}'.format(samples, prec, rec, f_1))

avg_prec = average_precision(pred_fs, Y)
avg_prec

# plot result of feature sum with normalization
plot_precision_recall_f1(sample_sizes, precs, recs, f1)


######## Random Forest ########

# Load dataset
X_train, X_test, Y_train, Y_test = split_train_test(data, test_size=0.3, mix_all=True)

print("n_est \t avrage precision")
for n_estimators in [10, 20, 50, 100, 150]:
    rf = RandomForestClassifier(n_estimators=n_estimators, max_features='auto', n_jobs=-1)
    rf.fit(X_train, Y_train)
    pred_rf = rf.predict_proba(X_test)[:, 1]
    avg_prec = average_precision(pred_rf, Y_test)
    print(n_estimators, '\t', avg_prec)

# By running a cross-validation, we find that n_estimators=100 gives the best performance.
# Choose this as the default model for the following work.
rf = RandomForestClassifier(n_estimators=100, max_features='auto', n_jobs=-1)
rf.fit(X_train, Y_train)

# Predicition, Calculate precision, recall, F1-score
pred_rf = rf.predict_proba(X_test)[:, 1]

sample_sizes = [8, 16, 32, 64, 128, 256, 512]
precs, recs, f1 = precision_recall(pred_rf, Y_test, *sample_sizes)

print("Total number of samples: {}\n".format(len(Y_test)))
print("{}\t{}\t{}\t{}".format('samples', 'precision', 'recall', 'f1_score'))
print('-'*40)
for samples, prec, rec, f_1 in zip(sample_sizes, precs, recs, f1):
    print('{}\t{:.3f}\t\t{:.3f}\t{:.3f}'.format(samples, prec, rec, f_1))

avg_prec = average_precision(pred_rf, Y_test)
avg_prec

# plot result of random forrest
plot_precision_recall_f1(sample_sizes, precs, recs, f1)


####### Feature selection #######

# List all features
features = list(X_train.columns)
print(features)

# According to the way by which they are generated, 129 features are divided in 5 categories:
# Localise, NetEMD, Basic, Path, Community Density.

categories_features = {
    "Localise": {'comb_abs90', 'comb_absolute_value', 'comb_exp_1', 'comb_exp_2', 'comb_exp_3', 'comb_exp_4',
                 'comb_ipr90', 'comb_ipr_1', 'comb_ipr_2', 'comb_ipr_3', 'comb_ipr_4', 'comb_sign_equal_1',
                 'comb_sign_equal_2', 'comb_sign_stat_1', 'comb_sign_stat_2', 'lower_abs90', 'lower_absolute_value',
                 'lower_exp_1', 'lower_exp_2', 'lower_exp_3', 'lower_exp_4', 'lower_ipr90', 'lower_ipr_1',
                 'lower_ipr_2', 'lower_ipr_3', 'lower_ipr_4', 'lower_sign_equal_1', 'lower_sign_equal_2',
                 'lower_sign_stat_1', 'lower_sign_stat_2', 'rw_abs90', 'rw_absolute_value', 'rw_exp_1', 'rw_exp_2',
                 'rw_exp_3', 'rw_exp_4', 'rw_ipr90', 'rw_ipr_1', 'rw_ipr_2', 'rw_ipr_3', 'rw_ipr_4', 'rw_sign_equal_1',
                 'rw_sign_equal_2', 'rw_sign_stat_1', 'rw_sign_stat_2', 'upper_abs90', 'upper_absolute_value',
                 'upper_exp_1', 'upper_exp_2', 'upper_exp_3', 'upper_exp_4', 'upper_ipr90', 'upper_ipr_1',
                 'upper_ipr_2', 'upper_ipr_3', 'upper_ipr_4', 'upper_sign_equal_1', 'upper_sign_equal_2',
                 'upper_sign_stat_1', 'upper_sign_stat_2'},
    "NetEMD": {'NetEMD_comb_1', 'NetEMD_comb_2', 'NetEMD_lower_1', 'NetEMD_lower_2', 'NetEMD_rw_1', 'NetEMD_rw_2',
               'NetEMD_upper_1', 'NetEMD_upper_2', 'in_out_strength_1', 'in_out_strength_2', 'in_strength_1',
               'in_strength_2', 'out_strength_1', 'out_strength_2', 'motif_10_1', 'motif_10_2', 'motif_11_1',
               'motif_11_2', 'motif_12_1', 'motif_12_2', 'motif_13_1', 'motif_13_2', 'motif_14_1', 'motif_14_2',
               'motif_15_1', 'motif_15_2', 'motif_16_1', 'motif_16_2', 'motif_4_1', 'motif_4_2', 'motif_5_1',
               'motif_5_2', 'motif_6_1', 'motif_6_2', 'motif_7_1', 'motif_7_2', 'motif_8_1', 'motif_8_2', 'motif_9_1',
               'motif_9_2'},
    "Basic": {'gaw10_score', 'gaw20_score', 'gaw_score'},
    "Path": {'path_10', 'path_11', 'path_12', 'path_13', 'path_14', 'path_15', 'path_16', 'path_17', 'path_18',
             'path_19', 'path_2', 'path_20', 'path_3', 'path_4', 'path_5', 'path_6', 'path_7', 'path_8', 'path_9'},
    "Com. Density": {'degree_std', 'first_density', 'first_strength', 'second_density', 'second_strength',
                     'small_community', 'third_density'}
}
features_categories = {}
for cat, feats in categories_features.items():
    for feat in feats:
        features_categories[feat] = cat

features = ['NetEMD_comb_1', 'NetEMD_comb_2', 'NetEMD_lower_1', 'NetEMD_lower_2', 'NetEMD_rw_1', 'NetEMD_rw_2',
            'NetEMD_upper_1', 'NetEMD_upper_2', 'comb_abs90', 'comb_absolute_value', 'comb_exp_1', 'comb_exp_2',
            'comb_exp_3', 'comb_exp_4', 'comb_ipr90', 'comb_ipr_1', 'comb_ipr_2', 'comb_ipr_3', 'comb_ipr_4',
            'comb_sign_equal_1', 'comb_sign_equal_2', 'comb_sign_stat_1', 'comb_sign_stat_2', 'degree_std',
            'first_density', 'first_strength', 'gaw10_score', 'gaw20_score', 'gaw_score', 'in_out_strength_1',
            'in_out_strength_2', 'in_strength_1', 'in_strength_2', 'lower_abs90', 'lower_absolute_value',
            'lower_exp_1', 'lower_exp_2', 'lower_exp_3', 'lower_exp_4', 'lower_ipr90', 'lower_ipr_1',
            'lower_ipr_2', 'lower_ipr_3', 'lower_ipr_4', 'lower_sign_equal_1', 'lower_sign_equal_2',
            'lower_sign_stat_1', 'lower_sign_stat_2', 'motif_10_1', 'motif_10_2', 'motif_11_1', 'motif_11_2',
            'motif_12_1', 'motif_12_2', 'motif_13_1', 'motif_13_2', 'motif_14_1', 'motif_14_2', 'motif_15_1',
            'motif_15_2', 'motif_16_1', 'motif_16_2', 'motif_4_1', 'motif_4_2', 'motif_5_1', 'motif_5_2', 'motif_6_1',
            'motif_6_2', 'motif_7_1', 'motif_7_2', 'motif_8_1', 'motif_8_2', 'motif_9_1', 'motif_9_2', 'out_strength_1',
            'out_strength_2', 'path_10', 'path_11', 'path_12', 'path_13', 'path_14', 'path_15', 'path_16', 'path_17',
            'path_18', 'path_19', 'path_2', 'path_20', 'path_3', 'path_4', 'path_5', 'path_6', 'path_7', 'path_8',
            'path_9', 'rw_abs90', 'rw_absolute_value', 'rw_exp_1', 'rw_exp_2', 'rw_exp_3', 'rw_exp_4', 'rw_ipr90',
            'rw_ipr_1', 'rw_ipr_2', 'rw_ipr_3', 'rw_ipr_4', 'rw_sign_equal_1', 'rw_sign_equal_2', 'rw_sign_stat_1',
            'rw_sign_stat_2', 'second_density', 'second_strength', 'small_community', 'third_density', 'upper_abs90',
            'upper_absolute_value', 'upper_exp_1', 'upper_exp_2', 'upper_exp_3', 'upper_exp_4', 'upper_ipr90',
            'upper_ipr_1', 'upper_ipr_2', 'upper_ipr_3', 'upper_ipr_4', 'upper_sign_equal_1', 'upper_sign_equal_2',
            'upper_sign_stat_1', 'upper_sign_stat_2']
len(features)

# Random Forest can assign each feature an importance
# Use this importance to sort features and select some most important ones.
feature_rank = sorted(zip(features, rf.feature_importances_), key=lambda x: x[1], reverse=True)
feature_rank

f_rank_cumsum = np.cumsum(np.array([x[1] for x in feature_rank]))
plt.plot(f_rank_cumsum)
plt.title("feature importance ranking")
plt.xlabel("number of ordered features")
plt.ylabel("cumulative feature importance")
plt.axvline(x=14, linestyle='-.', color='r')

selected_features = [feat for feat, feat_imp in feature_rank[:14]]
print(selected_features)

selected_features_categories = [features_categories[feat] for feat in selected_features]

plt.figure()
plt.hist(selected_features_categories, bins=4, align='mid', rwidth=0.9)
plt.title("Number of selected features in each category")
plt.ylabel("count of features")
plt.xlabel("feature categories")
plt.show()


####### Train model on selected features ########

# We then retrain the model on only the 14 selected features.
X_train = X_train.loc[:, selected_features]
X_test = X_test.loc[:, selected_features]

print("n_est \t avrage precision")
for n_estimators in [10, 20, 50, 100, 150]:
    rf = RandomForestClassifier(n_estimators=n_estimators, max_features='auto', n_jobs=-1)
    rf.fit(X_train, Y_train)
    pred_rf = rf.predict_proba(X_test)[:, 1]
    avg_prec = average_precision(pred_rf, Y_test)
    print(n_estimators, '\t', avg_prec)

rf = RandomForestClassifier(n_estimators=50, max_features='auto', n_jobs=-1)
rf.fit(X_train, Y_train)

pred_rf = rf.predict_proba(X_test)[:, 1]

sample_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
precs, recs, f1 = precision_recall(pred_rf, Y_test, *sample_sizes)

print("Total number of samples: {}\n".format(len(Y_test)))
print("{}\t{}\t{}\t{}".format('samples', 'precision', 'recall', 'f1_score'))
print('-'*40)
for samples, prec, rec, f_1 in zip(sample_sizes, precs, recs, f1):
    print('{}\t{:.3f}\t\t{:.3f}\t{:.3f}'.format(samples, prec, rec, f_1))

avg_prec = average_precision(pred_rf, Y_test)
avg_prec

plot_precision_recall_f1(sample_sizes, precs, recs, f1)


######## Test on whole networks ########

# Formerly, we mix all nodes of different networks together and split them for training and testing.
# In this section, we will try to train models on some networks and test them on some other networks.
# The testing networks are kept as a whole. Therefore, for each testing network,
# we perform the prediction and evaluation procedure,
# and then average the performance metrics on all testing networks.

# select_features is no longer None, to use only the 14 seleceted features
# mix_all set to False
X_train, X_test, Y_train, Y_test = split_train_test(data, test_size=0.3, mix_all=False, select_features=selected_features)

rf = RandomForestClassifier(n_estimators=50, max_features='auto', n_jobs=-1)
rf.fit(X_train, Y_train)

sample_sizes = [8, 16, 32, 64, 128, 256, 512]
performance_measures = ['precision', 'recall', 'f1-score', 'avg-precision']
network_index = sorted(list(range(len(X_test)))*len(sample_sizes))
performance_matrix = pd.DataFrame(columns=performance_measures, index=[network_index, sample_sizes*len(X_test)], dtype=float)
performance_matrix.index.names= ['network_id', 'sample_size']

for network_id, (network, labels) in enumerate(zip(X_test, Y_test)):
    pred = rf.predict_proba(network)[:, 1]
    avg_prec = average_precision(pred, labels)
    precs, recs, f1s = precision_recall(pred, labels, *sample_sizes)
    for sample_size, prec, rec, f1 in zip(sample_sizes, precs, recs, f1s):
        performance_matrix.loc[(network_id, sample_size), 'precision'] = prec
        performance_matrix.loc[(network_id, sample_size), 'recall'] = rec
        performance_matrix.loc[(network_id, sample_size), 'f1-score'] = f1
        performance_matrix.loc[(network_id, sample_size), 'avg-precision'] = avg_prec
performance_matrix

average_performance = performance_matrix.groupby('sample_size').mean()
std_performance = performance_matrix.groupby('sample_size').std()
average_performance

plt.figure(figsize=(9, 4))

plt.subplot(131)
plt.errorbar(sample_sizes, average_performance['precision'], yerr=std_performance['precision'], ecolor='r')
plt.xscale('log')
plt.title("precision")
plt.xlabel('No. nodes')

plt.subplot(132)
plt.errorbar(sample_sizes, average_performance['recall'], yerr=std_performance['recall'], ecolor='r')
plt.xscale('log')
plt.title("recall")
plt.xlabel('No. nodes')

plt.subplot(133)
plt.errorbar(sample_sizes, average_performance['f1-score'], yerr=std_performance['f1-score'], ecolor='r')
plt.xscale('log')
plt.title("F1-score")
plt.xlabel('No. nodes')

plt.tight_layout()

