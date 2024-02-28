import os
import json
import argparse

from src.eval.perplexity_eval import KNNEvaluator
from transformers import AutoTokenizer, AutoModelForCausalLM


class DataArgument:
  def __init__(self, data_type="text_repo", repo_path=None) -> None:
    self.type = data_type
    if repo_path is None:
      raise ValueError("repo_path must be provided")
    self.repo_path = repo_path
    
def arg_parse():
  parser = argparse.ArgumentParser(description='Evaluate model perplexity.')
  parser.add_argument('--model_name', type=str, default="Salesforce/codegen-350M-multi", help='Model name')
  parser.add_argument('--device', type=str, default='cuda:0', help='Device')
  parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
  parser.add_argument('--output_folder', type=str, default="/checkpoints/metric/", help='Output folder')
  parser.add_argument('--repos-folder', type=str, default="data/perplexity/repos", help='Repos folder')
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  
  args = arg_parse()
  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  model = AutoModelForCausalLM.from_pretrained(args.model_name)
  
  metric = {
    "codegen_train": 0.0,
    "knn_eval": 0.0,
    "codegen_eval": 0.0,
  }
  repo_dir = args.repos_folder
  # repo_dir = "/datadrive05/huypn16-backup/memorizing-coder/repos"
  save_metric_dir = args.output_folder
  
  num_of_repos = len(os.listdir(repo_dir))
  print("Number of repos to feed: ", num_of_repos)
  
  num_of_next = 0
  trailing = 0
  
  for repo in os.listdir(repo_dir):
    repo_path = os.path.join(repo_dir, repo)
    num_of_files = 0
    for dirpath, dirnames, filenames in os.walk(repo_path):
      num_of_files += 1
    
    data_args = DataArgument(
      repo_path=repo_path,
      data_type="text_repo"
    )
    if os.path.exists(os.path.join("checkpoints/datastore", repo)):
      os.system(f"rm -rf checkpoints/datastore/{repo}")
      os.mkdir(os.path.join("checkpoints/datastore", repo))
      
    evaluator = KNNEvaluator(
              model=model,
              dstore_dir=os.path.join("checkpoints/datastore", repo),
              tokenizer=tokenizer,
              device=args.device,
              batch_size=args.batch_size)
    
    try:
      evaluator.dataset_setup(data_args)
    except Exception as e:
      print(e)
      print("Error in dataset setup, skip this repo")
      num_of_next += 1
      continue
    
    evaluator.dataloader_setup()
    
    try:
      codegen_train_ppl = evaluator.run_eval_save_datastore()
    except Exception as e:
      print(e)
      print("Error in codegen train perplexity, skip this repo")
      num_of_next += 1
      continue
    
    knn_eval_ppl = evaluator.run_eval_knn_perplexity()
    codegen_eval_ppl = evaluator.run_eval_codegen_perplexity()
    trailing += 1
    
    metric["codegen_train"] += codegen_train_ppl
    metric["knn_eval"] += knn_eval_ppl
    metric["codegen_eval"] += codegen_eval_ppl
    
    print(f"At the repo {trailing}: ", metric["codegen_train"]/trailing, metric["knn_eval"]/trailing, metric["codegen_eval"]/trailing)
  
  metric["codegen_train"] /= (num_of_repos - num_of_next)
  metric["knn_eval"] /= (num_of_repos - num_of_next)
  metric["codegen_eval"] /= (num_of_repos - num_of_next)
  
  print("effectively evaluate on number of repos: ", num_of_repos - num_of_next)
  print(metric)
  
  with open(os.path.join(save_metric_dir, f"metric_{args.model_name}.json"), "w") as f:
    json.dump(metric, f)