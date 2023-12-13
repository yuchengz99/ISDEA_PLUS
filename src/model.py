import torch
from torch.nn import ModuleList
from torchdrug.layers import functional

class LinearEmbeddingModel(torch.nn.Module):
    def __init__(self,num_i,num_o):
        super(LinearEmbeddingModel,self).__init__()  
        self.linear1=torch.nn.Linear(num_i,num_o)

    def forward(self, x):
        x = self.linear1(x)
        return x

class SimpleEmbeddingModel(torch.nn.Module):
    def __init__(self,num_i,num_o):
        super(SimpleEmbeddingModel,self).__init__()
        
        self.linear1=torch.nn.Linear(num_i,num_o)
        self.act1=torch.nn.Tanh()
  
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        return x

class EmbeddingModel(torch.nn.Module):
    def __init__(self,num_i,num_h,num_o):
        super(EmbeddingModel,self).__init__()
        
        self.linear1=torch.nn.Linear(num_i,num_h)
        self.act1=torch.nn.Tanh()
        self.linear2=torch.nn.Linear(num_h,num_h) 
        self.act2=torch.nn.Tanh()
        self.linear3=torch.nn.Linear(num_h,num_o)
        self.act3=torch.nn.Tanh()
  
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        x = self.act3(x)
        return x
    
class LinkPredictionModel(torch.nn.Module):
    def __init__(self,num_i,num_h,num_o=1):
        super(LinkPredictionModel,self).__init__()
        
        self.linear1=torch.nn.Linear(num_i,num_h)
        self.relu1=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(num_h,num_h) 
        self.relu2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(num_h,num_o)
        self.sigmoid3=torch.nn.Sigmoid()
  
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.sigmoid3(x)
        return x

class FastISDEA(torch.nn.Module):
    def __init__(self, input_heuristics_dim, init_feature_dim, predictor_feature_hidden, sign_k, device):
        super(FastISDEA,self).__init__()

        self.sign_k = sign_k
        # self.device = device
        self.init_feature_dim = init_feature_dim

        self.heuristics_embedding = EmbeddingModel(input_heuristics_dim, init_feature_dim, init_feature_dim).to(device)

        self.mlp_1 = ModuleList()
        self.mlp_2 = ModuleList()
        self.mlp_3 = ModuleList()
        self.mlp_4 = ModuleList()
        for i in range(sign_k):
            self.mlp_1.append(LinearEmbeddingModel(init_feature_dim, init_feature_dim).to(device))
            self.mlp_2.append(LinearEmbeddingModel(init_feature_dim, init_feature_dim).to(device))
            self.mlp_3.append(LinearEmbeddingModel(init_feature_dim, init_feature_dim).to(device))
            self.mlp_4.append(LinearEmbeddingModel(init_feature_dim, init_feature_dim).to(device))

        output_feature_dim = init_feature_dim*(self.sign_k+1)
        self.mlp_r = EmbeddingModel(output_feature_dim, output_feature_dim, output_feature_dim).to(device)
        self.mlp_r_not = EmbeddingModel(output_feature_dim, output_feature_dim, output_feature_dim).to(device)

        self.combine = EmbeddingModel(output_feature_dim, int(output_feature_dim/2), int(output_feature_dim/2)).to(device)

        self.predictor = LinkPredictionModel(int(output_feature_dim*3/2)+init_feature_dim, predictor_feature_hidden, 1).to(device)

    def train_forward(self, input_relation_feature, input_node_feature, adj, adjs, heuristics, batch, chosen_relations, device):
        node_feature = self.process_node_feature(input_relation_feature, input_node_feature, adj, adjs, chosen_relations, device)

        label = []
        link_feature = []
        for edge in batch:
            if isinstance(edge[0], torch.Tensor):
                h = int(edge[0].item())
                t = int(edge[1].item())
                r = int(edge[2].item())
                v = edge[3].item()
            else:
                h = int(edge[0])
                t = int(edge[1])
                r = int(edge[2])
                v = edge[3]
            label.append(v)

            link = torch.cat((node_feature[r][h], node_feature[r][t], node_feature[r][h]*node_feature[r][t]))
            link_feature.append(link)
        label = torch.Tensor(label).reshape(len(label), 1).to(device)
        link_feature = torch.stack(link_feature, dim=0).to(device)
        heuristics_embedding = self.heuristics_embedding(heuristics).to(device)
        link_feature = torch.cat((link_feature, heuristics_embedding), 1)
        predicted = self.predictor(link_feature).to(device)
        return predicted, label
    
    def test_forward(self, x, node_feature, heuristics, negative_tail, negative_relation, device):
        link_feature = []
        for edge in x:
            if isinstance(edge[0], torch.Tensor): 
                h = int(edge[0].item())
                t = int(edge[1].item())
                r = int(edge[2].item())
            else:
                h = int(edge[0])
                t = int(edge[1])
                r = int(edge[2])

            link = torch.cat((node_feature[r][h], node_feature[r][t], node_feature[r][h]*node_feature[r][t]))
            link_feature.append(link)
        link_feature = torch.stack(link_feature, dim=0)
        heuristics_embedding = self.heuristics_embedding(heuristics).to(device)
        link_feature = torch.cat((link_feature, heuristics_embedding), 1)

        predicted = self.predictor(link_feature).to(device)
        batchsize = len(x) // (1 + negative_tail + negative_relation)

        rank = []
        for i in range(batchsize):
            output = []
            true_value = predicted[i*(1+negative_tail+negative_relation)][0].item()
            for j in range(i*(1+negative_tail+negative_relation),(i+1)*(1+negative_tail+negative_relation),1):
                output.append(predicted[j][0].item())

            output.sort(reverse=False)
            rank_1 = 1+negative_tail+negative_relation-output.index(true_value)
            output.sort(reverse=True)
            rank_2 = output.index(true_value)+1
            final_rank = (rank_1+rank_2)/2
            rank.append(final_rank)
        return rank
    
    def process_node_feature(self, input_relation_feature, input_node_feature, adj, adjs, chosen_relations, device):
        for i in range(len(chosen_relations)):
            if isinstance(chosen_relations[i], torch.Tensor): 
                chosen_relations[i] = chosen_relations[i].item()

        num_node = input_node_feature.size()[0]
        num_relation = len(adjs)

        adj = adj.to(device)

        relation_embedding = input_relation_feature
        initial_node_feature = input_node_feature

        feature_r = {}
        feature_r_not = {}
        final_node_feature = {}

        for chosen_r in chosen_relations:
            feature_r[chosen_r] = [initial_node_feature]
            feature_r_not[chosen_r] = [initial_node_feature]

        for i in range(self.sign_k):
            for chosen_r in chosen_relations:
                r_adj = adjs[chosen_r].to(device)
                r_not_adj = (adj - r_adj).to(device)

                feature_r_i = feature_r[chosen_r][i]
                feature_r_not_i = feature_r_not[chosen_r][i]

                if i == 0:
                    tmp_1 = self.mlp_3[i](feature_r_i)
                    tmp_2_rspmm = functional.generalized_rspmm(r_adj, relation_embedding, feature_r_i, sum="add", mul="mul")
                    tmp_2 = self.mlp_1[i](tmp_2_rspmm)
                    tmp_r = tmp_1 + tmp_2
                    feature_r[chosen_r].append(tmp_r)

                    tmp_1_not = self.mlp_4[i](feature_r_not_i)
                    tmp_2_not_rspmm = functional.generalized_rspmm(r_not_adj, relation_embedding, feature_r_not_i, sum="add", mul="mul")
                    tmp_2_not = self.mlp_2[i](tmp_2_not_rspmm)
                    tmp_r_not = tmp_1_not + tmp_2_not
                    feature_r_not[chosen_r].append(tmp_r_not)

                    # feature_r[chosen_r].append(self.mlp_3[i](feature_r_i) + self.mlp_1[i](functional.generalized_rspmm(r_adj, relation_embedding, feature_r_i, sum="add", mul="mul")))
                    # feature_r_not[chosen_r].append(self.mlp_4[i](feature_r_not_i) + self.mlp_2[i](functional.generalized_rspmm(r_not_adj, relation_embedding, feature_r_not_i, sum="add", mul="mul")))
                else:
                    feature_r[chosen_r].append(self.mlp_3[i](feature_r_i) + self.mlp_1[i](functional.generalized_rspmm(adj, relation_embedding, feature_r_i, sum="add", mul="mul")))
                    feature_r_not[chosen_r].append(self.mlp_4[i](feature_r_not_i) + self.mlp_2[i](functional.generalized_rspmm(adj, relation_embedding, feature_r_not_i, sum="add", mul="mul")))
                
        for chosen_r in chosen_relations:
            node_feature_r = torch.stack(feature_r[chosen_r], dim=1).reshape(num_node, -1)
            node_feature_r = self.mlp_r(node_feature_r)
            node_feature_r_not = torch.stack(feature_r_not[chosen_r], dim=1).reshape(num_node, -1)
            node_feature_r_not = self.mlp_r_not(node_feature_r_not)
            final_node_feature[chosen_r] = self.combine(node_feature_r + node_feature_r_not)

        return final_node_feature