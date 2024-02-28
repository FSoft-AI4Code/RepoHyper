import argparse
import torch
from tqdm import tqdm
import os
import pickle
import json

def args_parse():
    parser = argparse.ArgumentParser(description='Get retrieved results')
    parser.add_argument('--data-path', type=str, default='', help='extracted, labeled gnn data')
    parser.add_argument('--model', type=str, default='', help='model checkpoint path')
    parser.add_argument('--num_ctxt', type=int, default=8, help='number of context')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--train-size', type=float, default=0.8, help='train size')
    return parser.parse_args()

def main():
    train_json = []
    id = 0
    args = args_parse()
    data_path = args.data_path
    gnn_model = torch.load(args.model)
    gnn_model.eval()
    file_paths = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path)]
    for file_path in tqdm(file_paths):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        for sample in data:
            topk = []
            if (sample["gold_index"] is None) or (len(sample["embeddings"]) == 0) or (len(sample["edge_tensor"]) == 0) or (len(sample["query"]) == 0) or "import_indexes" not in sample.keys():
                continue
            logits = gnn_model(sample["embeddings"].to(args.device), sample["edge_tensor"].to(args.device), [0], sample["query"].to(args.device))
            logits = torch.functional.softmax(logits, dim=-1)
            import_logits = logits[sample["import_indexes"]]
            sorted, import_indices = torch.sort(import_logits, dim=0, descending=True)
            for i in range(args.num_ctxt):
                topk.append(sample["import_indexes"][import_indices[i]])
            for i in range(args.num_ctxt-len(topk)):
                highest_id = torch.topk(import_logits, 1, dim=0)[1][0]
                topk.append(sample["import_indexes"][highest_id])
                import_logits[highest_id] = -1
            data_point = {
                "id": id,
                "code": sample["code"],
                "target": sample["next_line"],
                "answers": [sample["next_line"]],
                "ctxs": [
                    {
                        "text": sample["kcge_context"][context_id]
                    }
                    for context_id in topk
                ]
            }
            id += 1
            train_json.append(data_point)
    with open("train.json", "w") as f:
        json.dump(train_json[:len(train_json)*args.train_size], f)
    with open("dev.json", "w") as f:
        json.dump(train_json[len(train_json)*args.train_size:], f)
        
if __name__ == "__main__":
    main()