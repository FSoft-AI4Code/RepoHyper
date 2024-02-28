import torch
from torch_geometric.nn import SAGEConv, to_hetero, GATv2Conv, GatedGraphConv
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Batch

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(SAGEConv(hidden_channels, hidden_channels, aggr="max"))
            # self.layers.append(GATv2Conv(hidden_channels, int(hidden_channels/4), heads=4, concat=True, dropout=0.0))
            # self.layers.append(GatedGraphConv(hidden_channels, num_layers=2))

        self.linear = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        
        x = self.linear(x)
        return x

class IdentityEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

    def forward(self, x, edge_index):
        return x
        
class RerankingDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, scoring_method="cosine"):
        if scoring_method not in ["cosine", "mlp"]:
            raise ValueError("scoring_method must be either 'cosine' or 'mlp'")
        super().__init__()
        self.scoring_method = scoring_method
        if self.scoring_method == "mlp":
            self.scorer = nn.Sequential(
                nn.Linear(hidden_channels*2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1)
            )
        elif self.scoring_method == "cosine":
            self.scorer = nn.Identity()
        
    def forward(self, batch):
        new_batches = []
        for idx, graph in enumerate(batch.to_data_list()):
            if self.scoring_method == "mlp":
                out = torch.concat([graph.x, graph.query.unsqueeze(0).repeat(graph.x.shape[0], 1)], dim=1)
                scores = self.scorer(out)
                setattr(graph, "scores", scores)
            elif self.scoring_method == "cosine":
                scores = torch.matmul(torch.nn.functional.normalize(graph.x, dim=1), torch.nn.functional.normalize(graph.query.squeeze(0), dim=0))
                setattr(graph, "scores", scores)
            new_batches.append(graph)
        return Batch.from_data_list(new_batches)

class SAGEReranker(torch.nn.Module):
    def __init__(self, hidden_channels, metadata=None, temp=0.01, num_enc_layers=3):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, num_layers=num_enc_layers)
        # self.encoder = IdentityEncoder(hidden_channels)
        self.decoder = RerankingDecoder(hidden_channels)
        # binary cross entropy loss with logits
        self.loss = torch.nn.functional.nll_loss
        self.temp = temp

    def forward(self, batch: Batch):
        embeddings = self.encoder(batch.x, batch.edge_index)
        batch.x = embeddings 
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
        top3_import_acc = 0
        top10_acc = 0
        top30_acc = 0
        top50_acc = 0
        for graph in batch.to_data_list():
            logits = F.softmax(graph.scores/self.temp, dim=0)
            top1_predict_idx = [graph.import_indexes[index] for index in torch.topk(logits[graph.import_indexes], k=1, dim=0).indices.tolist()]
            top1_predict_idx.extend(graph.index_to_name[top1_predict_idx[0]])
            top1_import_acc += (graph.y.item() in top1_predict_idx)
            
            top3_predict_idx = [graph.import_indexes[index] for index in torch.topk(logits[graph.import_indexes], k=(min(3, len(graph.scores[graph.import_indexes]))), dim=0).indices.tolist()]
            expanded_top3_predict_idx = []
            expanded_top3_predict_idx.extend(top3_predict_idx)
            for index in top3_predict_idx:
                expanded_top3_predict_idx.extend(graph.index_to_name[index])
            top3_import_acc += (graph.y.item() in top3_predict_idx)
            
            top10_acc += (graph.y.item() in torch.topk(logits, k=10, dim=0).indices)
            top30_acc += (graph.y.item() in torch.topk(logits, k=min(len(logits), 30), dim=0).indices)
            top50_acc += (graph.y.item() in torch.topk(logits, k=min(len(logits), 50), dim=0).indices)
        
        return top1_import_acc/len(batch), top3_import_acc/len(batch), top10_acc/len(batch), top30_acc/len(batch), top50_acc/len(batch)
            
