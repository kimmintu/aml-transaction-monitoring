import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
# %matplotlib inline

from rpy2.robjects.packages import importr
devtools = importr('devtools')
# devtools.install_github("dynverse/netdist", dependencies = True)
# devtools.install_github("alan-turing-institute/network-comparison")

from utils import generate_null_models, get_parameters
from generator import ER_generator, draw_anomalies
from basic_test import basic_features
from com_detection import community_detection
from spectral_localisation import spectral_features
from NetEMD import NetEMD_features
from path_finder import path_features

import json
from numpyencoder import NumpyEncoder
from networkx.readwrite import json_graph
from datetime import datetime
from sklearn.model_selection import train_test_split


num_models = 20     # original = 20
num_nodes = 1000    # original = 1000
num_basic_mc_samples = 300  # original = 500
num_references = 5     # original = 10
num_null_models = 12    # original = 60

ps = np.linspace(0.001, 0.05, 50)
ws = np.linspace(0.0, 0.01, 11)
candidate_parameters = get_parameters(num_nodes, ps, ws)
num_cand_param = len(candidate_parameters)

AML_TYPE_DICT = {None: 0, 'path': 1, 'star': 2, 'ring': 3, 'clique': 4, 'tree': 5}


def generate_feature_graph(model_id, p, w):
    # p, w = candidate_parameters[np.random.choice(range(num_cand_param))]
    logging.info("Computing {}-th/{} model (p={:.3f}, w={:.3f})".format(model_id, num_models, p, w))
    graph = ER_generator(n=num_nodes, p=p, seed=None)
    # graph = draw_anomalies(graph, w=1 - w)
    logging.info("\n\nGenerating null models 1\n\n")
    _, references = generate_null_models(graph, num_models=num_references, min_size=10)     # min_size=20 original
    logging.info("\n\nGenerating null models 2\n\n")
    null_samples_whole, null_samples = generate_null_models(graph, num_models=num_null_models, min_size=20)
    logging.info("\n\nGenerating NetEMD features\n\n")
    graph = NetEMD_features(graph, references, null_samples, num_references=num_references, num_samples=num_null_models)
    logging.info("\n\nGenerating basic features\n\n")
    graph = basic_features(graph, num_samples=num_basic_mc_samples)
    logging.info("\n\nGenerating community features\n\n")
    graph = community_detection(graph, null_samples, num_samples=20)
    logging.info("\n\nGenerating spectral features\n\n")
    graph = spectral_features(graph, null_samples, num_samples=num_null_models)
    logging.info("\n\nGenerating path features\n\n")
    graph = path_features(graph, null_samples_whole, num_samples=num_null_models)
    return graph


def write_json_graph(graph, model_id, p, w):
    data = json_graph.node_link_data(graph)
    with open('./data_fastgcn/input/Network_p_{:.3f}_w_{:.3f}_{}.json'.format(p, w, model_id), 'w') as outfile:
        json.dump(data, outfile, cls=NumpyEncoder)


def write_csv_df(graph, model_id, p, w):
    features = set()
    for node in graph.nodes():
        features |= set(graph.node[node].keys())
    # features.remove('type')
    logging.info("\n\nComposing DataFrame\n\n")
    X = pd.DataFrame.from_dict(dict(graph.nodes(data=True, default=0)), orient='index')
    X.fillna(0, inplace=True)
    X.replace([np.inf, -np.inf], 0, inplace=True)
    logging.info("\n\nWriting to local file\n\n")
    X.to_csv('./data_fastgcn/input/Network_p_{:.3f}_w_{:.3f}_{}.csv'.format(p, w, model_id))


def generate_multiple_graph_to_json_and_csv():
    for model_id in range(num_models):
        p, w = candidate_parameters[np.random.choice(range(num_cand_param))]
        graph = generate_feature_graph(model_id, p, w)
        write_json_graph(graph, model_id, p, w)
        write_csv_df(graph, model_id, p, w)


def generate_graph_dataset_json_for_fastgcn(model_id):
    p, w = candidate_parameters[np.random.choice(range(num_cand_param))]
    graph = generate_feature_graph(model_id, p, w)
    data = json_graph.node_link_data(graph)
    with open('data_run_test/big_graph_50k.json', 'w') as outfile:
        json.dump(data, outfile, cls=NumpyEncoder)


def create_class_map_json(G, path, file_name):
    class_map_json = {n: AML_TYPE_DICT[(G.node[n]).get("type", None)] for n in G.nodes()}
    with open(path + file_name + '-class_map.json', 'w') as outfile:
        json.dump(class_map_json, outfile, cls=NumpyEncoder)


def create_id_map_json(G, path, file_name):
    id_map_json = {n: ind for ind, n in enumerate(G.nodes())}
    with open(path + file_name + '-id_map.json', 'w') as outfile:
        json.dump(id_map_json, outfile, cls=NumpyEncoder)


def create_feats_npy(G, path, file_name):
    g_df = pd.DataFrame.from_dict(G.nodes, orient='index')
    g_df = g_df.fillna(0)
    g_df = g_df.drop('type', axis=1)
    feats = g_df.to_numpy()  # get df after removing type and index columns
    np.save(path + file_name + '-feats.npy', feats)


def create_train_val_test_graph(G, path, file_name):
    mapping = dict(zip(G.nodes(), map(str, G.nodes())))
    G = nx.relabel_nodes(G, mapping)

    class_map_json = {n: AML_TYPE_DICT[G[n].get("type", None)] for n in G.nodes()}
    x = list(class_map_json.keys())
    y = list(class_map_json.values())
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.2, random_state=0, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=.50, random_state=0, stratify=y_train)

    for n in G.nodes():
        G.node[n]['test'] = False
        G.node[n]['val'] = False
    for n in x_train:
        G.node[n]['test'] = True
    for n in x_val:
        G.node[n]['val'] = True

    data = json_graph.node_link_data(G)
    with open(path + file_name + '-updated.json', 'w') as outfile:
        json.dump(data, outfile, cls=NumpyEncoder)


def standard_graph_to_multiple_datasource(path, file_name):
    G = json_graph.node_link_graph(json.load(open(path + file_name + '.json')))
    create_class_map_json(G, path, file_name)
    create_id_map_json(G, path, file_name)
    create_feats_npy(G, path, file_name)
    create_train_val_test_graph(G, path, file_name)


if __name__=="__main__":
    start = datetime.now()
    print('starting...................................: ', start)

    generate_multiple_graph_to_json_and_csv()
    # generate_graph_dataset_json_for_fastgcn(10)
    # standard_graph_to_multiple_datasource('data_fastgcn/input/', 'Network_p_0.016_w_0.003_1')

    end = datetime.now()
    print('starting...................................: ', start)
    print('finish.....................................: ', end)
    print('duration...................................: ', (end - start))

