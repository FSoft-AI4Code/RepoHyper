from utils import load_data
import os
from joblib import Parallel, delayed
from tqdm import tqdm


language = "python"
task = "retrieval"
settings = ["cross_file_first"]

cross_file_first_easy = load_data(task, language, "cross_file_first")["train"]["easy"]
cross_file_first_hard = load_data(task, language, "cross_file_first")["train"]["hard"]
cross_file_first = cross_file_first_hard + cross_file_first_easy

unique_repo_names = set()

for sample in cross_file_first:
    unique_repo_names.add(sample["repo_name"])

unique_repo_names = list(unique_repo_names)

def get_all_samples_from_repo(repo_name, dataset):
    samples = []
    for sample in dataset:
        if sample["repo_name"] == repo_name:
            samples.append(sample)
    return samples

def download_repo(repo):
    file_name = repo.split("/")[-1]
    if file_name not in os.listdir("data/repobench/repos/"):
        os.system(f'git clone --depth 1 --single-branch https://github.com/{repo} data/repobench/repos/{file_name}')
    else:
        print(f"Already downloaded {repo}")

Parallel(n_jobs=40, prefer="threads")(
    delayed(download_repo)(name) for name in tqdm(unique_repo_names))
