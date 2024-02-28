import os
from subprocess import check_call, DEVNULL, STDOUT
from joblib import Parallel, delayed
from tqdm import tqdm
import json

REPOS_FOLDER = "data/repobench/repos/"
REPOS_TRANSLATED_FOLDER = "/datadrive05/huypn16/knn-transformers/data/repobench/repos_translated"

def get_py_files(target_folder):
    relative_file_paths = []

    # use os.walk to go through each file in the target path and its sub-folders
    for root, dirs, files in os.walk(target_folder):
        for file in files:
            # check if the file is a Python file
            if file.endswith(".py"):
                # get the full file path
                full_file_path = os.path.join(root, file)
                # get the relative file path by replacing the target path from the full file path
                relative_file_path = full_file_path.replace(target_folder, '')
                # append the relative file path to the list
                if relative_file_path.startswith("/"):
                    relative_file_path = relative_file_path[1:]
                relative_file_paths.append(relative_file_path)
    
    return relative_file_paths

def check_compile(py_file, repo_name):
    command_str = f"cd data/repobench/repos/{repo_name} && python3 -m py_compile {py_file}"
    returned_value = os.system(command_str)
    if returned_value == 0:
        return True
    else:
        return False

def generate_call_graphs(repo_name):
    py_files = get_py_files(os.path.join(REPOS_FOLDER, repo_name))
    for py_file in py_files:
        if not check_compile(py_file, repo_name):
            py_files.remove(py_file)
    # command_str = f"cd data/repobench/repos/{repo_name} && python3 -m pycg  {' '.join(py_files)} --max-iter 3 -o ../../repos_call_graphs/{repo_name}.json > /dev/null 2>&1"
    # os.system(command_str)
    
    # Move to the required directory
    print(repo_name)
    os.chdir(f'/datadrive05/huypn16/knn-transformers/data/repobench/repos/{repo_name}')

    # Run the python3 command
    command_str = f"python3 -m pycg  {' '.join(py_files)} --max-iter 3 -o ../../repos_call_graphs/{repo_name}.json"
    success = os.system(command_str)
    if success != 0:
        if not os.path.exists(f'/datadrive05/huypn16/knn-transformers/data/repobench/repos_translated/{repo_name}'):
            os.mkdir(f'/datadrive05/huypn16/knn-transformers/data/repobench/repos_translated/{repo_name}')
        success_translation = os.system(f'2to3 --output-dir=/datadrive05/huypn16/knn-transformers/data/repobench/repos_translated/{repo_name} -W -n /datadrive05/huypn16/knn-transformers/data/repobench/repos/{repo_name}')
        print(success_translation)
        py_files = get_py_files(os.path.join(REPOS_TRANSLATED_FOLDER, repo_name))
        if success_translation == 0:
            os.chdir(f'/datadrive05/huypn16/knn-transformers/data/repobench/repos_translated/{repo_name}')
            command_str = f"python3 -m pycg  {' '.join(py_files)} --max-iter 3 -o ../../repos_call_graphs/{repo_name}.json"
            os.system(command_str)
def main():
    repo_names = os.listdir(REPOS_FOLDER)
    created_names = [name.removesuffix(".json") for name in os.listdir("data/repobench/repos_call_graphs")]
    non_empty_created_names = []
    for created_name in created_names:
        with open(f"data/repobench/repos_call_graphs/{created_name}.json", "r") as f:
            data = json.load(f)
        if len(data) != 0:
            non_empty_created_names.append(created_name)
    repo_names = list(set(repo_names) - set(non_empty_created_names))
    Parallel(n_jobs=30, prefer="threads")(
        delayed(generate_call_graphs)(name) for name in tqdm(repo_names))
    # for name in repo_names[:1]:
    #     generate_call_graphs(name)
    
if __name__ == "__main__":
    main()