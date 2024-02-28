import os
import time
import pickle

import torch
import torch.nn.functional as F

from src.models.GCN import SAGEReranker
from src.data.dataset_gnn import RepoBenchData
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import argparse
import wandb

def argument_parsing():
    parser = argparse.ArgumentParser(description="Train GNN")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=5e-5, help="Weight decay")
    parser.add_argument("--num-epochs", type=int,
                        default=20, help="Number of epochs")
    parser.add_argument("--local_rank", type=int,
                        default=-1, help="Local rank")
    parser.add_argument("--nfeat", type=int, default=768,
                        help="Number of features")
    parser.add_argument("--temp", type=float, default=0.01, help="Temperature of Softmax")
    parser.add_argument("--layers", type=int, default=1,
                        help="Number of Encoder layers")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--data-path", type=str,
                        default="data/repobench/repos_graphs_labeled_cosine_radius_unix", help="Data path")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--output", type=str,
                        default="checkpoints/gnn", help="Output path")
    parser.add_argument("--train-size", type=float, default=0.8, help="Train size")
    parser.add_argument("--dataset-cache-path", type=str, default="data/repobench/cache/dataset.pkl", help="Dataset cache path")
    return parser


if __name__ == "__main__":
    logger = wandb.init(project="gnn-link-prediction")
    args = argument_parsing().parse_args()
    device = torch.device(args.device)
    model = SAGEReranker(
        hidden_channels=args.nfeat, 
        metadata=None, 
        temp=args.temp,
        num_enc_layers=args.layers
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if not os.path.exists(args.dataset_cache_path):
        trainset, valset = RepoBenchData(args.data_path).split(args.train_size)
        print("Saving dataset cache...at {}".format(args.dataset_cache_path))
        with open(args.dataset_cache_path, "wb") as f:
            pickle.dump((trainset, valset), f)
    else:
        with open(args.dataset_cache_path, "rb") as f:
            trainset, valset = pickle.load(f)
            print("Loading dataset cache...from {}".format(args.dataset_cache_path))
            
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, 
                              follow_batch=["x", "edge_index", "query", "y", "ignore_index", "index_to_name", "import_indexes"])
    
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=True, 
                            follow_batch=["x", "edge_index", "query", "y", "import_indexes", "index_to_name"], 
                            exclude_keys=["ignore_index"])
    
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}")
        for batch in tqdm(train_loader):
            loss = model.train(batch.cuda())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logger.log({"loss": loss.item()})
        
        print("Evaluating...")
        with torch.no_grad():
            top1_import_acc = 0
            top3_import_acc = 0
            top10_acc = 0
            top30_acc = 0
            top50_acc = 0
            for batch in tqdm(val_loader):
                batch_top1_import_acc, batch_top3_import_acc, batch_top10_acc, batch_top30_acc, batch_top50_acc = model.eval(batch.cuda())
                top1_import_acc += batch_top1_import_acc
                top3_import_acc += batch_top3_import_acc
                top10_acc += batch_top10_acc
                top30_acc += batch_top30_acc
                top50_acc += batch_top50_acc
                
            print(f"Top1 import accuracy: {top1_import_acc/len(val_loader)}")
            print(f"Top3 import accuracy: {top3_import_acc/len(val_loader)}")
            print(f"Top10 accuracy: {top10_acc/len(val_loader)}")
            print(f"Top30 accuracy: {top30_acc/len(val_loader)}")
            print(f"Top50 accuracy: {top50_acc/len(val_loader)}")
            
            logger.log({"top1_import_acc": top1_import_acc/len(val_loader)})
            logger.log({"top3_import_acc": top3_import_acc/len(val_loader)})
            logger.log({"top10_acc": top10_acc/len(val_loader)})
            logger.log({"top30_acc": top30_acc/len(val_loader)})
            logger.log({"top50_acc": top50_acc/len(val_loader)})
        
        torch.save(model.state_dict(), os.path.join(args.output, f"model_{epoch}.pt"))
