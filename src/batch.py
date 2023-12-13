import numpy as np
import networkx as nx
import os
import torch
from torch.utils.data import DataLoader
import random

def batch(edge_list, num_rel, batchsize):
    relation_edge = []
    for i in range(num_rel):
        relation_edge.append([])
    for edge in edge_list:
        relation_edge[edge[2]].append(np.array(edge))
    data = []
    for i in range(num_rel):
        if len(relation_edge[i]) > 0:
            relation_loader = DataLoader(relation_edge[i], batch_size=batchsize, shuffle=True)
            for edge_batch in relation_loader:
                if edge_batch is not None:
                    data.append(edge_batch)
    return data

def negative_sampling(positive_edge_batch, head_rel2tail, head_tail2rel, num_node, num_relation, negative_tail=2, negative_relation=2, mode="test", full_graph=False):
    if isinstance(positive_edge_batch, torch.Tensor): 
        positive_edge_batch = positive_edge_batch.tolist()
    mixed_batch = []
    for positive_edge in positive_edge_batch:
        head = positive_edge[0]
        tail = positive_edge[1]
        true_relation = positive_edge[2]
        if true_relation >= num_relation:
            return []
        
        false_relation = []
        if not full_graph:
            while len(false_relation) < negative_relation:
                r = random.randint(0, num_relation-1)
                if num_relation - 1 >= negative_relation:
                    if r != true_relation and r not in false_relation and r not in head_tail2rel[(head, tail)]:
                    # if [head, tail, r] not in edge_list:
                        false_relation.append(r)
                else:
                    if r != true_relation and r not in head_tail2rel[(head, tail)]:
                    # if [head, tail, r] not in edge_list:
                        false_relation.append(r)
        elif negative_relation > 0:
            false_relation = list(range(num_relation))
            false_relation.remove(true_relation)
            for r in head_tail2rel[(head, tail)]:
                false_relation.remove(r)

        false_tail = []
        if not full_graph:
            while len(false_tail) < negative_tail:
                t = random.randint(0, num_node-1)
                if t not in head_rel2tail[(head, true_relation)] and t not in false_tail:
                    false_tail.append(t)
                # if len(G_list) == 2:
                #     if t not in G_list[0].neighbors(head) and t not in G_list[1].neighbors(head) and t not in false_tail:
                #         false_tail.append(t)
                # if len(G_list) == 3:
                #     if t not in G_list[0].neighbors(head) and t not in G_list[1].neighbors(head) and t not in G_list[2].neighbors(head) and t not in false_tail:
                #         false_tail.append(t)
        elif negative_tail > 0:
            false_tail = list(v for v in range(num_node) if v not in head_rel2tail[(head, true_relation)])

        positive_edge.append(1)
        if mode == "train":
            mixed_batch.append(positive_edge)
            # for _ in range(negative_relation + negative_tail):
            #     mixed_batch.append(positive_edge)
        if mode == "test":
            mixed_batch.append(positive_edge)
        for t in false_tail:
            mixed_batch.append([head, t, true_relation, 0])
        for r in false_relation:
            mixed_batch.append([head, tail, r, 0])
    false_relation.append(true_relation)
    return mixed_batch
