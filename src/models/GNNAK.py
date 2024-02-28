import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import src.models.pyg_gnn_wrapper as gnn_wrapper 
from src.models.utils import MLP, DiscreteEncoder, Identity, VNUpdate
from src.models.PPGN import PPGN
from torch_geometric.nn.inits import reset
from torch_geometric.nn import to_hetero
from src.data.dataset_gnn import HeRepoBenchData
import torch_geometric.transforms as T
from src.data.transform import SubgraphsTransform
from torch_geometric.loader import DataLoader
from src.models.he_GCN import HeLinkPredictionDecoder
from torch_geometric.data import Batch

BN = True

class GNN(nn.Module):
    # this version use nin as hidden instead of nout, resulting a larger model
    def __init__(self, nin, nout, nlayer, gnn_type, dropout=0, bn=BN, bias=True, res=True):
        super().__init__()
        # TODO: consider remove input and output encoder for nhead=1?
        # self.input_encoder = MLP(nin, nin, nlayer=2, with_final_activation=True) #if nin!=nout else nn.Identity()
        self.convs = nn.ModuleList([getattr(gnn_wrapper, gnn_type)(nin, nin, bias=not bn) for _ in range(nlayer)]) # set bias=False for BN
        self.norms = nn.ModuleList([nn.BatchNorm1d(nin) if bn else Identity() for _ in range(nlayer)])
        self.output_encoder = MLP(nin, nout, nlayer=1, with_final_activation=False, bias=bias) if nin!=nout else Identity() 
        self.dropout = dropout
        self.res = res

    def reset_parameters(self):
        # self.input_encoder.reset_parameters()
        self.output_encoder.reset_parameters()
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            norm.reset_parameters()
     
    def forward(self, x, edge_index, edge_attr, batch):
        # x = self.input_encoder(x)
        previous_x = x
        for layer, norm in zip(self.convs, self.norms):
            x = layer(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x = x + previous_x 
                previous_x = x

        x = self.output_encoder(x)
        return x

class SubgraphGNNKernel(nn.Module):
    # Use GNN to encode subgraphs
    # gnn_types: a list of GNN types
    def __init__(self, nin, nout, nlayer, gnn_types, dropout=0, 
                       hop_dim=16, 
                       bias=True, 
                       res=True,
                       pooling='mean',
                       embs=(0,1,2),
                       embs_combine_mode='add',
                       mlp_layers=1,
                       subsampling=False, 
                       online=True):
        super().__init__()
        assert max(embs) <= 2 and min(embs) >= 0
        assert embs_combine_mode in ['add', 'concat']

        use_hops = hop_dim > 0
        nhid = nout // len(gnn_types)
        self.hop_embedder = nn.Embedding(20, hop_dim)
        self.gnns = nn.ModuleList()
        for gnn_type in gnn_types:
            if gnn_type == 'PPGN':
                gnn = PPGN(nin+hop_dim if use_hops else nin, nhid, nlayer)
            else:
                gnn = GNN(nin+hop_dim if use_hops else nin, nhid, nlayer, gnn_type, dropout=dropout, res=res)
            self.gnns.append(gnn)
        self.subgraph_transform = MLP(nout, nout, nlayer=mlp_layers, with_final_activation=True)
        self.context_transform =  MLP(nout, nout, nlayer=mlp_layers, with_final_activation=True)

        self.out_encoder = MLP(nout if embs_combine_mode=='add' else nout*len(embs), nout, nlayer=mlp_layers, 
                               with_final_activation=False, bias=bias, with_norm=True)

        self.use_hops = use_hops
        self.gate_mapper_subgraph = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid())
        self.gate_mapper_context = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid())
        self.gate_mapper_centroid = nn.Sequential(nn.Linear(hop_dim, nout), nn.Sigmoid()) # add this one to scale centroid embedding
        self.subsampling = subsampling

        # for correctly counting number of parameters
        # self.centroid_transform = Identity()
        # self.subgraph_transform = Identity()
        # self.context_transform = Identity()
        # self.gate_mapper_context = Identity()
        # self.gate_mapper_centroid = Identity()
        # self.out_encoder = Identity()

        # dropout = 0
        self.dropout = dropout
        self.online = online
        self.pooling = pooling
        self.embs = embs
        self.embs_combine_mode = embs_combine_mode

    def reset_parameters(self):
        self.hop_embedder.reset_parameters()
        for gnn in self.gnns:
            gnn.reset_parameters()
        self.subgraph_transform.reset_parameters()
        self.context_transform.reset_parameters()
        self.out_encoder.reset_parameters()
        reset(self.gate_mapper_context)
        reset(self.gate_mapper_subgraph)
        reset(self.gate_mapper_centroid)

    def forward(self, data):
        # prepare x, edge_index, edge_attr for the combined subgraphs
        combined_subgraphs_x = data.x[data.subgraphs_nodes_mapper] # lift up the embeddings, positional encoding
        combined_subgraphs_edge_index = data.combined_subgraphs
        combined_subgraphs_edge_attr = data.edge_attr[data.subgraphs_edges_mapper]
        combined_subgraphs_batch = data.subgraphs_batch
        if self.use_hops:
            hop_emb = self.hop_embedder(data.hop_indicator+1)  # +1 to make -1(not calculated part: too far away) as 0. 
            combined_subgraphs_x = torch.cat([combined_subgraphs_x, hop_emb], dim=-1)

        combined_subgraphs_x = torch.cat([gnn(combined_subgraphs_x, combined_subgraphs_edge_index, combined_subgraphs_edge_attr, combined_subgraphs_batch)
                                          for gnn in self.gnns], dim=-1) # -1 dim = nout
        # for correctly counting number of parameters
        # subgraph_x = combined_subgraphs_x
        # subgraph_x = subgraph_x * self.gate_mapper_subgraph(hop_emb)
        # x = scatter(subgraph_x, combined_subgraphs_batch, dim=0, reduce=self.pooling)
        
        if self.subsampling and self.training:
            centroid_x_selected = combined_subgraphs_x[(data.subgraphs_nodes_mapper == data.selected_supernodes[combined_subgraphs_batch])]
            subgraph_x_selected = self.subgraph_transform(F.dropout(combined_subgraphs_x, self.dropout, training=self.training)) if len(self.embs) > 1 else combined_subgraphs_x
            context_x_selected = self.context_transform(F.dropout(combined_subgraphs_x, self.dropout, training=self.training)) if len(self.embs) > 1 else combined_subgraphs_x
            if self.use_hops:
                centroid_x_selected = centroid_x_selected * self.gate_mapper_centroid(hop_emb[(data.subgraphs_nodes_mapper == data.selected_supernodes[combined_subgraphs_batch])]) 
                subgraph_x_selected = subgraph_x_selected * self.gate_mapper_subgraph(hop_emb)
                context_x_selected = context_x_selected * self.gate_mapper_context(hop_emb)
            subgraph_x_selected = scatter(subgraph_x_selected, combined_subgraphs_batch, dim=0, reduce=self.pooling)
            context_x_selected = scatter(context_x_selected, data.subgraphs_nodes_mapper, dim=0, reduce=self.pooling)

            # propagate features from selected nodes to non-selected nodes, used as an estimator
            centroid_x = data.x.new_zeros((len(data.x), centroid_x_selected.size(-1)))
            centroid_x[data.selected_supernodes] = centroid_x_selected
            subgraph_x = data.x.new_zeros((len(data.x), subgraph_x_selected.size(-1)))
            subgraph_x[data.selected_supernodes] = subgraph_x_selected  
            for i in range(1, data.edges_between_two_hops.max()+1):
                # print((new.sum(-1)==0).sum(), (data.hops_to_selected==i).sum() )
                bipartite = data.edge_index[:, data.edges_between_two_hops==i]
                scatter(centroid_x[bipartite[0]], bipartite[1], dim=0, reduce='mean', out=centroid_x)
                scatter(subgraph_x[bipartite[0]], bipartite[1], dim=0, reduce='mean', out=subgraph_x)
            
            # scale up the context embedding when using add pooling
            context_x = context_x_selected * data.subsampling_scale.unsqueeze(-1) if self.pooling == 'add' else context_x_selected
        else:
            centroid_x = combined_subgraphs_x[(data.subgraphs_nodes_mapper == combined_subgraphs_batch)]
            subgraph_x = self.subgraph_transform(F.dropout(combined_subgraphs_x, self.dropout, training=self.training)) if len(self.embs) > 1 else combined_subgraphs_x
            context_x = self.context_transform(F.dropout(combined_subgraphs_x, self.dropout, training=self.training)) if len(self.embs) > 1 else combined_subgraphs_x
            if self.use_hops:
                centroid_x = centroid_x * self.gate_mapper_centroid(hop_emb[(data.subgraphs_nodes_mapper == combined_subgraphs_batch)]) 
                subgraph_x = subgraph_x * self.gate_mapper_subgraph(hop_emb)
                context_x = context_x * self.gate_mapper_context(hop_emb)
            subgraph_x = scatter(subgraph_x, combined_subgraphs_batch, dim=0, reduce=self.pooling)
            context_x = scatter(context_x, data.subgraphs_nodes_mapper, dim=0, reduce=self.pooling)

        x = [centroid_x, subgraph_x, context_x]
        x = [x[i] for i in self.embs]
        if self.embs_combine_mode == 'add':
            x = sum(x)
        else:
            x = torch.cat(x, dim=-1)
            # last part is only essential for embs_combine_mode = 'concat', can be ignored when overfitting
            x = self.out_encoder(F.dropout(x, self.dropout, training=self.training)) 
            
        return x


class GNNAsKernel(nn.Module):
    def __init__(self, nfeat_node, nfeat_edge, nhid, nout, nlayer_outer, nlayer_inner, gnn_types, 
                        dropout=0, 
                        hop_dim=0, 
                        node_embedding=False, 
                        use_normal_gnn=False, 
                        bn=BN, 
                        vn=False, 
                        res=True, 
                        pooling='mean',
                        embs=(0,1,2),
                        embs_combine_mode='add',
                        mlp_layers=1,
                        subsampling=False, 
                        online=True):
        super().__init__()
        # special case: PPGN
        nlayer_ppgn = nlayer_inner
        if gnn_types[0] == 'PPGN' and nlayer_inner == 0:
            # normal PPGN
            nlayer_ppgn = nlayer_outer
            nlayer_outer = 1

        # nfeat_in is None: discrete input features
        self.input_encoder = DiscreteEncoder(nhid) if nfeat_node is None else MLP(nfeat_node, nhid, 1)
        # layers
        edge_emd_dim = nhid if nlayer_inner == 0 else nhid + hop_dim 
        self.edge_encoders = nn.ModuleList([DiscreteEncoder(edge_emd_dim) if nfeat_edge is None else MLP(nfeat_edge, edge_emd_dim, 1)
                                            for _ in range(nlayer_outer)])

        self.subgraph_layers = nn.ModuleList([SubgraphGNNKernel(nhid, nhid, nlayer_inner, gnn_types, dropout, 
                                                                hop_dim=hop_dim, 
                                                                bias=not bn,
                                                                res=res, 
                                                                pooling=pooling,
                                                                embs=embs,
                                                                embs_combine_mode=embs_combine_mode,
                                                                mlp_layers=mlp_layers,
                                                                subsampling=subsampling, 
                                                                online=online) 
                                              for _ in range(nlayer_outer)])
                                              
        self.norms = nn.ModuleList([nn.BatchNorm1d(nhid) if bn else Identity() for _ in range(nlayer_outer)])
        self.output_decoder = MLP(nhid, nout, nlayer=2, with_final_activation=False)
        
        # traditional layers: for comparison 
        if gnn_types[0] != 'PPGN':
            self.traditional_gnns = nn.ModuleList([getattr(gnn_wrapper, gnn_types[0])(nhid, nhid, bias=not bn) for _ in range(nlayer_outer)])
            # For correctly counting number of parameters
            # self.traditional_gnns = nn.ModuleList(Identity() for _ in range(nlayer_outer))
        else:
            self.traditional_gnns = nn.ModuleList(PPGN(nhid, nhid, nlayer_ppgn) for _ in range(nlayer_outer))
       # virtual node
        self.vn_aggregators = nn.ModuleList([VNUpdate(nhid) for _ in range(nlayer_outer)])
        # For correctly counting number of parameters
        # self.vn_aggregators = nn.ModuleList(Identity() for _ in range(nlayer_outer))

        # record params
        self.gnn_type = gnn_types[0]
        self.dropout = dropout
        self.num_inner_layers = nlayer_inner
        self.node_embedding = node_embedding
        self.use_normal_gnn = use_normal_gnn
        self.hop_dim = hop_dim
        self.vn = vn
        self.res = res
        self.pooling = pooling


    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        self.output_decoder.reset_parameters()
        for edge_encoder, layer, norm, old, vn in zip(self.edge_encoders, self.subgraph_layers, self.norms, self.traditional_gnns, self.vn_aggregators):
            edge_encoder.reset_parameters()
            layer.reset_parameters()
            norm.reset_parameters()
            old.reset_parameters()
            vn.reset_parameters()


    def forward(self, data):
        x = data.x 
        x = self.input_encoder(x)
        ori_edge_attr = data.edge_attr 
        if ori_edge_attr is None:
            ori_edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))

        previous_x = x # for residual connection
        virtual_node = None
        for i, (edge_encoder, subgraph_layer, normal_gnn, norm, vn_aggregator) in enumerate(zip(self.edge_encoders, 
                                        self.subgraph_layers, self.traditional_gnns, self.norms, self.vn_aggregators)):
            # if ori_edge_attr is not None: # Encode edge attr for each layer
            data.edge_attr = edge_encoder(ori_edge_attr) 
            data.x = x
            if self.num_inner_layers == 0: 
                # standard message passing nn 
                if self.gnn_type == 'PPGN':
                    x = normal_gnn(data.x, data.edge_index, data.edge_attr, data.batch)
                else:
                    x = normal_gnn(data.x, data.edge_index, data.edge_attr)
            else:
                if self.use_normal_gnn:
                    if self.gnn_type == 'PPGN':
                        x = subgraph_layer(data) + normal_gnn(data.x, data.edge_index, data.edge_attr[:,:-self.hop_dim], data.batch)
                    else:
                        x = subgraph_layer(data) + normal_gnn(data.x, data.edge_index, data.edge_attr[:,:-self.hop_dim])
                else:
                    x = subgraph_layer(data)

            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

            if self.vn:
                virtual_node, x = vn_aggregator(virtual_node, x, data.batch)

            if self.res:
                x = x + previous_x
                previous_x = x # for residual connection

        # if not self.node_embedding:
        #     x = scatter(x, data.batch, dim=0, reduce=self.pooling)
            
        x = F.dropout(x, self.dropout, training=self.training)  
        x = self.output_decoder(x) 
        return x 
    
class GraphKernelLinker(torch.nn.Module):
    def __init__(self, hidden_channels, metadata=None, temp=0.01):
        super().__init__()
        self.encoder = to_hetero(
            GNNAsKernel(
                nfeat_node=hidden_channels,
                nfeat_edge=None,
                nhid=hidden_channels,
                nout=hidden_channels,
                nlayer_outer=4,
                nlayer_inner=2,
                gnn_types=['GINEConv'],
                embs_combine_mode='add',
            ),
            metadata=metadata
        )
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
        

if __name__ == "__main__":
    dataset = HeRepoBenchData(
        folder_path="data/repobench/repos_graphs_labeled_link_with_called_imported_edges",
        transforms=[
            T.ToUndirected(), 
            SubgraphsTransform(
                hops=2,
                sampling_mode=None
            )
        ]
    )
    sample_model = GraphKernelLinker(768, dataset.metadata)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch in data_loader:
        batch = batch.to(sample_model.device)
        print(sample_model(batch))
        break