import numpy as np
import networkx as nx
import os
import torch
from torch.utils.data import DataLoader
import random
import math

def load_dict(dict_path):
    dic = {}
    with open(dict_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            if line[1] not in dic:
                dic[line[1]] = int(line[0])
    return dic

def load_graph(edge_path, entity_dict, relation_dict, mode):
    data = {}
    num_node = len(entity_dict)
    num_relation = len(relation_dict) * 2

    if mode == "supervision":
        edge_list = []

        G = nx.Graph()
        for j in range(num_node):
            G.add_node(j)

        with open(edge_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                h = entity_dict[line[0]]
                r = relation_dict[line[1]]
                t = entity_dict[line[2]]
                G.add_edge(h, t)
                edge_list.append([h, t, r])
        
        data["G"] = G
        data["edge_list"] = edge_list

    if mode == "both":
        threshold = 0.1

        edge_list_mes = []
        edge_list_sup = []

        G_mes = nx.Graph()
        G_mes_di = nx.DiGraph()
        G_sup = nx.Graph()

        for j in range(num_node):
            G_mes.add_node(j)
            G_mes_di.add_node(j)
            G_sup.add_node(j)

        with open(edge_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                h = entity_dict[line[0]]
                r = relation_dict[line[1]]
                t = entity_dict[line[2]]
                r_reverse = int(r + num_relation/2)
                if random.random() < threshold:
                    G_sup.add_edge(h, t)
                    edge_list_sup.append([h, t, r])
                    edge_list_sup.append([t, h, r_reverse])
                else:
                    G_mes.add_edge(h, t)
                    G_mes_di.add_edge(h, t)

                    edge_list_mes.append([h, t, r])
                    edge_list_mes.append([t, h, r_reverse])
        
        i = [[], [], []]
        v = []

        I = []
        V = []
        for _ in range(num_relation):
            I.append([[], [], []])
            V.append([])

        E_h = torch.zeros(num_node, num_relation)
        E_t = torch.zeros(num_node, num_relation)

        for edge in edge_list_mes:
            h = edge[0]
            t = edge[1]
            r = edge[2]

            i[0].append(h)
            i[1].append(t)
            i[2].append(r)
            v.append(float(1.0/math.sqrt(G_mes.degree(h)*G_mes.degree(t))))

            E_h[h][r] += 1
            E_t[t][r] += 1

            I[r][0].append(h)
            I[r][1].append(t)
            I[r][2].append(r)
            V[r].append(float(1.0/math.sqrt(G_mes.degree(h)*G_mes.degree(t))))

        adjs = []
        for j in range(num_relation):
            adjs.append(torch.sparse_coo_tensor(I[j], V[j], (num_node , num_node, num_relation)))
        adj = torch.sparse_coo_tensor(i, v, (num_node, num_node, num_relation))

        data["G_mes"] = G_mes
        data["G_mes_di"] = G_mes_di
        data["G_sup"] = G_sup
        data["adj"] = adj
        data["adjs"] = adjs
        data["edge_list_mes"] = edge_list_mes
        data["edge_list_sup"] = edge_list_sup
        data["E_h"] = E_h
        data["E_t"] = E_t
        data["num_node"] = num_node
        data["num_relation"] = num_relation

    if mode == "message-passing":
        edge_list = []

        G = nx.Graph()
        G_di = nx.DiGraph()
        for j in range(num_node):
            G.add_node(j)
            G_di.add_node(j)

        with open(edge_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                h = entity_dict[line[0]]
                r = relation_dict[line[1]]
                t = entity_dict[line[2]]
                G.add_edge(h, t)
                G_di.add_edge(h, t)

                edge_list.append([h, t, r])
                r_reverse = int(r + num_relation/2)
                edge_list.append([t, h, r_reverse])

        i = [[], [], []]
        v = []

        I = []
        V = []
        for _ in range(num_relation):
            I.append([[], [], []])
            V.append([])

        E_h = torch.zeros(num_node, num_relation)
        E_t = torch.zeros(num_node, num_relation)

        for edge in edge_list:
            h = edge[0]
            t = edge[1]
            r = edge[2]

            i[0].append(h)
            i[1].append(t)
            i[2].append(r)
            v.append(float(1.0/math.sqrt(G.degree(h)*G.degree(t))))

            E_h[h][r] += 1
            E_t[t][r] += 1

            I[r][0].append(h)
            I[r][1].append(t)
            I[r][2].append(r)
            V[r].append(float(1.0/math.sqrt(G.degree(h)*G.degree(t))))
        
        adjs = []
        for j in range(num_relation):
            adjs.append(torch.sparse_coo_tensor(I[j], V[j], (num_node , num_node, num_relation)))
        adj = torch.sparse_coo_tensor(i, v, (num_node, num_node, num_relation))

        data["G"] = G
        data["G_di"] = G_di
        data["adj"] = adj
        data["adjs"] = adjs
        data["edge_list"] = edge_list
        data["E_h"] = E_h
        data["E_t"] = E_t
        data["num_node"] = num_node
        data["num_relation"] = num_relation
    return data

def load_data(name, data_folder, mode="train"):
    assert mode in ["train", "test"]

    data_folder = f"{data_folder}/{name}-trans" if mode == "train" else f"{data_folder}/{name}-ind"

    entities_path = os.path.join(data_folder, "entities.dict")
    relations_path = os.path.join(data_folder, "relations.dict")

    if mode == "train":
        msg_path = os.path.join(data_folder, "train.txt")
        sup_path = os.path.join(data_folder, "valid.txt")
    else:
        msg_path = os.path.join(data_folder, "observe.txt")
        sup_path = os.path.join(data_folder, "test.txt")

    entities_dict = load_dict(entities_path)
    relations_dict = load_dict(relations_path)

    if mode == "train":
        msg_data = load_graph(msg_path, entities_dict, relations_dict, "both")
    else:
        msg_data = load_graph(msg_path, entities_dict, relations_dict, "message-passing")
    
    sup_data = load_graph(sup_path, entities_dict, relations_dict, "supervision")

    return msg_data, sup_data

    # if len(data_folder) == 1:
    #     train_folder = data_folder[0] + "{name}-trans/".format(name=name)
    #     inf_folder = data_folder[0] + "{name}-ind/".format(name=name)
    
    # if len(data_folder) == 2:
    #     train_folder = data_folder[0]
    #     inf_folder = data_folder[1]

    # train_trainG = train_folder + "/train.txt"
    # train_validG = train_folder + "/valid.txt"
    # train_entity = train_folder + "/entities.dict"
    # train_relation = train_folder + "/relations.dict"

    # inf_observeG = inf_folder + "observe.txt"
    # inf_testG =  inf_folder + "test.txt"
    # inf_entity = inf_folder + "entities.dict"
    # inf_relation = inf_folder + "relations.dict"

    # train_entity_dict = load_dict(train_entity)
    # train_relation_dict = load_dict(train_relation)
    # train_train_data = load_graph(train_trainG, train_entity_dict, train_relation_dict, "both")
    # train_valid_data = load_graph(train_validG, train_entity_dict, train_relation_dict, "supervision")

    # inf_entity_dict = load_dict(inf_entity)
    # inf_relation_dict = load_dict(inf_relation)
    # inf_observe_data = load_graph(inf_observeG, inf_entity_dict, inf_relation_dict, "message-passing")
    # inf_test_data = load_graph(inf_testG, inf_entity_dict, inf_relation_dict, "supervision")

    # return train_train_data, train_valid_data, inf_observe_data, inf_test_data