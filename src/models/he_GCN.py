import torch
from torch_geometric.nn import SAGEConv, to_hetero, norm
from torch_geometric.nn.conv import HGTConv, FastHGTConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroDictLinear
from torch_geometric.data import Batch
from copy import deepcopy
import random

def expand_cls(predict_idx, index_to_name):
    expand_predict_idx = [predict_idx[0]]
    for index, name in index_to_name.items():
        if name.split(".")[-1] in index_to_name[predict_idx[0]]:
            expand_predict_idx.append(index)
    return list(set(expand_predict_idx))

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers=3, metadata=None):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                # SAGEConv(hidden_channels, hidden_channels, aggr="max", dropout=0.6)
                HGTConv(hidden_channels, hidden_channels, metadata, kwargs={"aggr":"max"})
            )

    def forward(self, x_dict, edge_index_dict):
        for i, layer in enumerate(self.layers):
            x_dict = layer(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) if key != "code" else x for key, x in x_dict.items()}
        return x_dict 
    
class HeRerankingDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.scorer = nn.Identity()
        self.scoring_method = "cosine"
        
    def forward(self, batch):
        new_batches = []
        for idx, graph in enumerate(batch.to_data_list()):
            if self.scoring_method == "mlp":
                out = torch.concat([graph.x, graph.query.unsqueeze(0).repeat(graph.x.shape[0], 1)], dim=1)
                scores = self.scorer(out)
                setattr(graph, "scores", scores)
            elif self.scoring_method == "cosine":
                scores = torch.zeros(sum([len(indexes) for indexes in graph.map_indexes.values()])).to(graph.x_dict["functions"].device)
                for type in ["functions", "classes", "code"]:
                    type_node_score = torch.matmul(torch.nn.functional.normalize(graph.x_dict[type], dim=1), torch.nn.functional.normalize(graph.query.squeeze(0), dim=0))
                    scores[graph.map_indexes[type]] = type_node_score
                setattr(graph, "scores", scores)
            new_batches.append(graph)
        return Batch.from_data_list(new_batches)

class HeLinkPredictionDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels*2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        self.pair_normalizer = norm.PairNorm(scale_individually=True)
        
    def forward(self, batch):
        new_batches = []
        for idx, graph in enumerate(batch.to_data_list()):
            graph.x_dict["functions"] = torch.nn.functional.normalize(graph.x_dict["functions"], dim=1)
            graph.x_dict["classes"] = torch.nn.functional.normalize(graph.x_dict["classes"], dim=1)
            target_node = graph.x_dict["functions"][-1]
            for type in ["functions", "classes"]:
                if type == "functions":
                    out_func = torch.cat([graph.x_dict[type][:-1], target_node.repeat(graph.x_dict[type][:-1].shape[0], 1)], dim=1)
                elif type == "classes":
                    out_cls = torch.cat([graph.x_dict[type], target_node.repeat(graph.x_dict[type].shape[0], 1)], dim=1)
                    
            out = torch.cat([out_func, out_cls], dim=0)
            scores = self.mlp(out)
            setattr(graph, "scores", scores)
            new_batches.append(graph)
        return Batch.from_data_list(new_batches)
            
class HGTReranker(torch.nn.Module):
    def __init__(self, hidden_channels, metadata=None, temp=0.01, num_enc_layers=2):
        super().__init__()
        self.encoder = to_hetero(GNNEncoder(hidden_channels, num_layers=num_enc_layers), metadata=metadata)
        self.decoder = HeLinkPredictionDecoder(hidden_channels)
        # binary cross entropy loss with logits
        self.loss = torch.nn.functional.nll_loss
        self.temp = temp

    def forward(self, batch: Batch):
        embeddings = self.encoder(batch.x_dict, batch.edge_index_dict)
        # workaround for batch.x_dict = embeddings can no pass the gradient
        new_batches = []
        running_idx_types = {"functions": 0, "classes": 0, "code": 0}
        for graph in batch.to_data_list():
            new_x_dict = {}
            for type in ["functions", "classes", "code"]:
                new_x_dict[type] = embeddings[type][running_idx_types[type] : running_idx_types[type] + graph.x_dict[type].shape[0]]
                running_idx_types[type] += graph.x_dict[type].shape[0]
            graph.x_dict = new_x_dict
            new_batches.append(graph)
        batch = Batch.from_data_list(new_batches)
        out = self.decoder(batch)
        return out
    
    def train(self, batch):
        loss = 0
        batch = self.forward(batch)
        for graph in batch.to_data_list():
            logits = F.log_softmax(graph.scores/self.temp, dim=0)
            # weight = torch.ones(graph.scores.shape[0])
            # weight[graph.ignore_index.unsqueeze(0)] = 0
            loss += self.loss(logits.unsqueeze(0), target=graph.y)
        return loss/len(batch)

    def eval(self, batch):
        batch = self.forward(batch)
        top1_import_acc = 0
        top10_acc = 0
        top30_acc = 0
        top50_acc = 0
        for graph in batch.to_data_list():
            logits = F.log_softmax(graph.scores/self.temp, dim=0)
            predict_idx = [graph.import_indexes[index] for index in torch.topk(logits[graph.import_indexes], k=1, dim=0).indices.tolist()]
            top1_import_acc += (graph.y.item() in predict_idx)
            top10_acc += (graph.y.item() in torch.topk(logits, k=10, dim=0).indices)
            top30_acc += (graph.y.item() in torch.topk(logits, k=min(len(logits), 30), dim=0).indices)
            top50_acc += (graph.y.item() in torch.topk(logits, k=min(len(logits), 50), dim=0).indices)
        
        return top1_import_acc/len(batch), top10_acc/len(batch), top30_acc/len(batch), top50_acc/len(batch)
        
class HeSAGELinker(torch.nn.Module):
    def __init__(self, hidden_channels, metadata=None, temp=0.01, num_enc_layers=2):
        super().__init__()
        # self.encoder = to_hetero(GNNEncoder(hidden_channels, num_layers=num_enc_layers), metadata=metadata)
        self.encoder = GNNEncoder(hidden_channels, num_layers=num_enc_layers, metadata=metadata)
        self.decoder = HeLinkPredictionDecoder(hidden_channels)
        # binary cross entropy loss with logits
        self.loss = torch.nn.functional.binary_cross_entropy
        self.temp = temp

    def forward(self, batch: Batch):
        embeddings = self.encoder(batch.x_dict, batch.edge_index_dict)
        # workaround for batch.x_dict = embeddings can no pass the gradient
        new_batches = []
        running_idx_types = {"functions": 0, "classes": 0, "code": 0}
        for graph in batch.to_data_list():
            new_x_dict = {}
            for type in ["functions", "classes", "code"]:
                new_x_dict[type] = embeddings[type][running_idx_types[type] : running_idx_types[type] + graph.x_dict[type].shape[0]]
                running_idx_types[type] += graph.x_dict[type].shape[0]
            graph.x_dict = new_x_dict
            new_batches.append(graph)
        batch = Batch.from_data_list(new_batches)
        out = self.decoder(batch)
        return out
    
    def train(self, batch):
        loss = 0
        batch = self.forward(batch)
        for graph in batch.to_data_list():
            logits = torch.sigmoid(graph.scores)
            labels = torch.zeros(graph.scores.shape[0])
            labels[graph.y] = 1
            graph.import_indexes.remove(graph.y)
            hard_examples = graph.import_indexes
            easy_examples = random.sample(torch.where(labels == 0)[0].tolist(), 5)
            selected_logits = torch.cat([logits[hard_examples], logits[easy_examples], logits[graph.y]], dim=0)
            selected_labels = torch.cat([labels[hard_examples], labels[easy_examples], labels[graph.y.cpu()]], dim=0)
            loss += self.loss(selected_logits.squeeze(-1).to(logits.device), target=selected_labels.to(logits.device))
        return loss/len(batch)

    # def train(self, batch):
    #     loss = 0
    #     batch = self.forward(batch)
    #     for graph in batch.to_data_list():
    #         logits = torch.sigmoid(graph.scores)
    #         print(logits)
    #         labels = torch.zeros(graph.scores.shape[0])
    #         labels[graph.y] = 1
    #         loss += self.loss(logits.squeeze(-1).to(logits.device), target=labels.to(logits.device))
    #     return loss/len(batch)
    
    def eval(self, batch):
        batch = self.forward(batch)
        top1_import_acc = 0
        top10_acc = 0
        top30_acc = 0
        top50_acc = 0
        for graph in batch.to_data_list():
            logits = torch.sigmoid(graph.scores.reshape(-1))
            predict_idx = [graph.import_indexes[index] for index in torch.topk(logits[graph.import_indexes], k=1, dim=0).indices.tolist()]
            top1_import_acc += (graph.y.item() in predict_idx)
            top10_acc += (graph.y.item() in torch.topk(logits, k=min(len(logits),10), dim=0).indices)
            top30_acc += (graph.y.item() in torch.topk(logits, k=min(len(logits), 30), dim=0).indices)
            top50_acc += (graph.y.item() in torch.topk(logits, k=min(len(logits), 50), dim=0).indices)
        
        return top1_import_acc/len(batch), top10_acc/len(batch), top30_acc/len(batch), top50_acc/len(batch)

if __name__ == "__main__":
    from src.data.dataset_gnn import HeRepoBenchData
    from torch_geometric.loader import DataLoader
    dataset = HeRepoBenchData("data/repobench/repos_graphs_labeled_cosine_radius_unix_link")
    model = HGTReranker(768, metadata=dataset.metadata)
    train_loader = DataLoader(dataset, batch_size=2)
    for batch in train_loader:
        loss = model.train(batch)
        