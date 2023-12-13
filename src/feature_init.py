import torch
import numpy as np
import networkx as nx
from tqdm import tqdm

def glorot_init(input_feature):
    # gain = torch.nn.init.calculate_gain('relu')
    return torch.nn.init.xavier_normal_(input_feature, gain = 1)

def normalization(feature):
    a = min(feature)
    b = max(feature)
    c = sum(feature)/len(feature)
    d = np.std(feature)
    feature = [ float((x-c)/(d)) for x in feature]
    return feature

def relation_feature_init(E_h, E_t):
    feature_dim = 5
    num_relation = E_h.size()[1]
    num_node = E_h.size()[0]

    relation_feature = torch.zeros(feature_dim, num_relation).tolist()

    degree = torch.sum(E_h, 0).tolist()

    h_h = torch.mm(E_h.T, E_h)
    h_t = torch.mm(E_h.T, E_t)
    t_h = torch.mm(E_t.T, E_h)
    t_t = torch.mm(E_t.T, E_h)

    h_h = torch.where(h_h > 0, 1, 0)
    h_t = torch.where(h_t > 0, 1, 0)
    t_h = torch.where(t_h > 0, 1, 0)
    t_t = torch.where(t_t > 0, 1, 0)

    h_h_feature = torch.sum(h_h, 0).tolist()
    h_t_feature = torch.sum(h_t, 0).tolist()
    t_h_feature = torch.sum(t_h, 0).tolist()
    t_t_feature = torch.sum(t_t, 0).tolist()

    degree = normalization(degree)
    h_h_feature = normalization(h_h_feature)
    h_t_feature = normalization(h_t_feature)
    t_h_feature = normalization(t_h_feature)
    t_t_feature = normalization(t_t_feature)
    relation_feature[0] = degree
    relation_feature[1] = h_h_feature
    relation_feature[2] = h_t_feature
    relation_feature[3] = t_h_feature
    relation_feature[4] = t_t_feature
    relation_feature = torch.Tensor(relation_feature).T
    return relation_feature

def node_feature_init(G):
    feature_dim = 6
    n = G.number_of_nodes()
    initial_feature = torch.zeros(feature_dim, n).tolist()

    # degree feature 
    degree_feature = []
    for idx, node in enumerate(G.nodes()):
        degree_feature.append(G.degree(node))
    degree_feature = normalization(degree_feature)
    initial_feature[0] = degree_feature

    # egonet feature
    egonet_within = []
    egonet_without = []
    n_edges = len(G.edges())
    for idx, node in enumerate(G.nodes()):
        ego_graph = nx.ego_graph(G, node, radius=1)
        n_within_edges = len(ego_graph.edges())
        n_external_edges = n_edges - n_within_edges
        egonet_within.append(n_within_edges)
        egonet_without.append(n_external_edges)
    egonet_within = normalization(egonet_within)
    egonet_without = normalization(egonet_without)
    initial_feature[1] = egonet_within
    initial_feature[2] = egonet_without

    # triangle feature
    triangles = nx.triangles(G)
    triangle_feature = []
    for idx, node in enumerate(G.nodes()):
        triangle_feature.append(triangles[node])
    triangle_feature = normalization(triangle_feature)
    initial_feature[3] = triangle_feature

    # k-core feature
    G.remove_edges_from(nx.selfloop_edges(G))
    kcore = nx.core_number(G)
    kcore_feature = []
    for idx, node in enumerate(G.nodes()):
        kcore_feature.append(kcore[node])
    kcore_feature = normalization(kcore_feature)
    initial_feature[4] = kcore_feature

    # clique feature
    cn = nx.node_clique_number(G)
    clique_feature = []
    for idx, node in enumerate(G.nodes()):
        clique_feature.append(cn[node])
    clique_feature = normalization(clique_feature)
    initial_feature[5] = clique_feature

    initial_feature = torch.Tensor(initial_feature).T
    return initial_feature

def heuristics_init(G, G_di, dis, batch):
    heuristics = []
    for edge in batch:
        if isinstance(edge[0], torch.Tensor): 
            h = int(edge[0].item())
            t = int(edge[1].item())
        else:
            h = int(edge[0])
            t = int(edge[1])
        
        feature = []

        # cn = nx.common_neighbors(G, h, t)
        # jaccard = nx.jaccard_coefficient(G, [(h, t)])
        # ra = nx.resource_allocation_index(G, [(h, t)])
        # aa = nx.adamic_adar_index(G, [(h, t)])
        
        if h not in dis:
            dis[h] = {}
            dis[h][t] = G.number_of_nodes()
        else:
            if t not in dis[h]:
                dis[h][t] = G.number_of_nodes()
        if t not in dis:
            dis[t] = {}
            dis[t][h] = G.number_of_nodes()
        else:
            if h not in dis[t]:
                dis[t][h] = G.number_of_nodes()

        h_t = float(dis[h][t]/G.number_of_nodes())
        t_h = float(dis[t][h]/G.number_of_nodes())
        feature.append(h_t)
        feature.append(t_h)
        
        # cn_count = 0
        # for _ in cn:
        #     cn_count += 1
        # feature.append(cn_count)

        # for u, v, p in jaccard:
        #     jaccard_index = p
        # feature.append(jaccard_index)

        # for u, v, p in ra:
        #     ra_index = p
        # feature.append(ra_index)

        # for u, v, p in aa:
        #     aa_index = p
        # feature.append(aa_index)
        heuristics.append(torch.Tensor(feature))
    heuristics = torch.stack(heuristics, dim=0)
    return heuristics