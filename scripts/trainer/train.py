import os
import time
import pickle

import torch
import torch.nn.functional as F

from src.models.GAT import GAT
from src.train.common import TrainerParams, ModelParams, RepoBenchData
  
class Trainer:
  def __init__(self, model_params, data, trainer_params) -> None:
    self.model = GAT(
      model_params.nfeat, 
      model_params.nhid, 
      model_params.dropout, 
      model_params.alpha, 
      model_params.nheads, 
      model_params.nlayers
    )
    self.trainer_params = trainer_params
    self.device = trainer_params.device
    self.model.to(self.device)
    self.optimizer = torch.optim.Adam(
      self.model.parameters(), 
      lr=trainer_params.lr, 
      weight_decay=trainer_params.weight_decay
    )
    self.data = data
    self.loss = torch.nn.NLLLoss().to(self.device)
    
  def train_epoch(self):
    print("Training...")
    total_graphs = 0
    for batch_id in range(len(self.data)):
      batch = self.data[batch_id]
      if batch is None:
        continue
      batch.to(self.device)
      self.model.train()
      self.optimizer.zero_grad()
      idx_train = batch.idx_train
      batched_logits = self.model(
        x=batch.embedding, 
        adj=batch.adj,
        index_train=idx_train,
        q=batch.query
      )
      loss = 0.0
      for sample_idx, sample_st_idx in enumerate(idx_train):
        sample_end_idx = idx_train[sample_idx + 1] if sample_idx + 1 < len(idx_train) else len(batch.adj)
        sample_logsoftmax = F.log_softmax(batched_logits[sample_st_idx:sample_end_idx], dim=0)
        sample_st_idx = sample_end_idx
        # print(sample_logsoftmax.shape)
        # print(sample_logsoftmax)
        label_id = batch.labels[sample_idx]
        loss += self.loss(sample_logsoftmax.unsqueeze(0), label_id.unsqueeze(0))
      total_graphs += len(idx_train)
      loss.backward()
      self.optimizer.step()
    print("Total graphs:", total_graphs)
    return float(loss)

  @torch.no_grad()
  def test(self):
    self.model.eval()
    acc_5 = 0
    acc_10 = 0
    acc_15 = 0
    acc_30 = 0
    acc_50 = 0
    num_valid_batch = 0
    for batch_id in range(len(self.data)):
      batch = self.data[batch_id]
      if batch is None:
        continue
      num_valid_batch += 1
      batch.to(self.device)
      batched_logits = self.model(
        x=batch.embedding, 
        adj=batch.adj,
        index_train=batch.idx_train,
        q=batch.query
      )
      batch_acc_5 = 0
      batch_acc_10 = 0
      batch_acc_15 = 0
      batch_acc_30 = 0
      batch_acc_50 = 0
      for sample_idx, sample_st_idx in enumerate(batch.idx_train):
        sample_end_idx = batch.idx_train[sample_idx + 1] if sample_idx + 1 < len(batch.idx_train) else len(batch.adj)
        # print(len(batched_logits[sample_st_idx:sample_end_idx]))
        if len(batched_logits[sample_st_idx:sample_end_idx]) > 5:
          pred_top_5 = torch.topk(batched_logits[sample_st_idx:sample_end_idx], k=5, dim=0)
        if len(batched_logits[sample_st_idx:sample_end_idx]) > 10:
          pred_top_10 = torch.topk(batched_logits[sample_st_idx:sample_end_idx], k=10, dim=0)
        if len(batched_logits[sample_st_idx:sample_end_idx]) > 15:
          pred_top_15 = torch.topk(batched_logits[sample_st_idx:sample_end_idx], k=15, dim=0)
        if len(batched_logits[sample_st_idx:sample_end_idx]) > 30:
          pred_top_30 = torch.topk(batched_logits[sample_st_idx:sample_end_idx], k=30, dim=0)
        if len(batched_logits[sample_st_idx:sample_end_idx]) > 50:
          pred_top_50 = torch.topk(batched_logits[sample_st_idx:sample_end_idx], k=50, dim=0)
        
        sample_st_idx = sample_end_idx
        batch_acc_5 += (batch.labels[sample_idx] in pred_top_5.indices)
        batch_acc_10 += (batch.labels[sample_idx] in pred_top_10.indices)
        batch_acc_15 += (batch.labels[sample_idx] in pred_top_15.indices)
        batch_acc_30 += (batch.labels[sample_idx] in pred_top_30.indices)
        batch_acc_50 += (batch.labels[sample_idx] in pred_top_50.indices)
      
      acc_5 += batch_acc_5 / len(batch.idx_train)
      acc_10 += batch_acc_10 / len(batch.idx_train)
      acc_15 += batch_acc_15 / len(batch.idx_train)
      acc_30 += batch_acc_30 / len(batch.idx_train)
      acc_50 += batch_acc_50 / len(batch.idx_train)
    return (acc_5 / num_valid_batch, acc_10 / num_valid_batch, acc_15 / num_valid_batch, acc_30 / num_valid_batch, acc_50 / num_valid_batch)

  def train(self):
    times = []
    best_val_acc = final_test_acc = 0
    for epoch in range(1, self.trainer_params.num_epochs + 1):
        start = time.time()
        loss = self.train_epoch()
        acc_10, acc_30, acc_50, acc_100, acc_120 = self.test()
        print("Epoch:", epoch, "Loss:", loss, "Test", acc_10, acc_30, acc_50, acc_100, acc_120)
        times.append(time.time() - start)
    print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")  

if __name__ == "__main__":
  trainer_params = TrainerParams(lr=0.005, weight_decay=5e-4, num_epochs=300, device=torch.device('cuda:0'))
  model_params = ModelParams(nfeat=768, nhid=96, dropout=0.6, alpha=0.2, nheads=8)
  data = RepoBenchData("/datadrive05/huypn16/knn-transformers/data/repobench/repos_graphs_labeled")
  trainer = Trainer(model_params, data, trainer_params)
  trainer.train()
  # print(data_params[0]['embedding'].shape)
  # print(data_params[0]['query'].shape)
  # print(data_params[0]['adj'].shape)
  # print(data_params[0]['idx_train'])
  # print(data_params[0]['labels'])