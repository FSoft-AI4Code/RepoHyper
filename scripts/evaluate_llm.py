# Python
import torch
from torch.utils.data import DataLoader
import argparse
import os
from src.utils import load_data
from src.metrics import calc_metrics
from src.llm import LLMModel

language = "python"
task = "pipeline"
settings = ["cross_file_first"]

cross_file_first_hard = load_data(task, language, "cross_file_first")["test"]["hard"]

def evaluate_model(model_name, retrieved_contexts):
    # Load the model
    model = LLMModel(model_name)
    model.eval()

    total_metric = 0

    # Iterate over the data
    for (batch, contexts),  in zip(cross_file_first_hard.split(8), retrieved_contexts.split(8)):
        inputs, targets = batch["code"], batch["next_line"]

        # Forward pass
        outputs = model.complete(inputs, contexts)

        # Compute accuracy
        correct = calc_metrics(outputs, targets)
        total_metric += correct

    avg_accuracy = total_metric / len(cross_file_first_hard)

    print(f'Average Metric: {avg_accuracy}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM model')
    parser.add_argument('--data', type=str, required=True, help='Path to the data')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers')

    args = parser.parse_args()

    data_path = args.data
    model_name = args.model
    num_workers = args.num_workers

    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist.")
        return
    retrieved_contexts = load_data(args.data, language, "retrieved_contexts")
    evaluate_model(model_name, retrieved_contexts)

if __name__ == "__main__":
    main()