<div align="center">



# RepoHyper: Better Context Retrieval Is All You Need for Repository-Level Code Completion
[![arXiv](https://img.shields.io/badge/arXiv-2305.06156-b31b1b.svg)](https://arxiv.org/abs/2403.06095)

</div>

## Introduction

We introduce RepoHyper, an novel framework transforming code completion into a seamless end-to-end process for use case on real world repositories. Traditional approaches depend on integrating contexts into Code Language Models (CodeLLMs), often presuming these contexts to be inherently accurate. However, we've identified a gap: the standard benchmarks don't always present relevant contexts.

To address this, RepoHyper proposes in three novel steps:

- Construction of a Code Property Graph, establishing a rich source of context.
- A novel Search Algorithm for pinpointing the exact context needed.
- The Expand Algorithm, designed to uncover implicit connections between code elements (akin to the Link Prediction problem on social network mining).

Our comprehensive evaluations reveal that RepoHyper sets a new standard, outperforming other strong baseline on the RepoBench benchmark.

## Installation

```bash
pip install -r requirements.txt
```

## Architecture
<img src="arch.png" width="750" height="350">

RepoHyper is a two-stage model. The first stage is a search-then-expand algorithm on Repo-level Semantic Graph (RSG) then use GNN link predictor that reranks the retrieved results from KNN search and graph expansion. The second stage is any code LLM model that takes the retrieved context and predicts the next line of code.

## Checkpoints
We provide the checkpoints for the GNN model [here](https://ai4code.blob.core.windows.net/repohyper/model_10.pt). The GNN model is trained on the RepoBench-R dataset with gold labels. We also provide [RepoBench-R RGSs](https://ai4code.blob.core.windows.net/repohyper/repos_graphs_labeled_link_with_called_imported_edges) to reproduce the results.


## Usage

### Data preparation

We need to clone [Repobench dataset](https://github.com/Leolty/repobench/tree/main/data) into `data/repobench` folder. Then download all the unique repositories used in this dataset

```bash
python3 -m scripts.data.download_repos --dataset data/repobench --output data/repobench/repos --num-processes 8
```

The next step is to generate call graph using PyCG. We use the following command to generate call graph for each repository. 60 processes are used to speed up the process (maximum RAM usage is around 350GB).

```bash
python3 -m scripts.data.generate_call_graph --repos data/repobench/repos --output data/repobench/repos_call_graphs --num-processes 60
```

Now we need to generate embeddings for each node for node embedding as well as create adjacency matrix by aligning Tree-sitter functions, classes, methods with call graph nodes. 
```bash
python3 -m scripts.data.repo_to_embeddings --repos data/repobench/repos --call-graphs data/repobench/repos_call_graphs --output data/repobench/repos_graphs --num-processes 60
```

Final step is labeling which node is the most optimal for predicting next line using gold snippet from repobench dataset. In this step, we also generate the training data for GNN training by extracting the subgraph using KNN search and RSG expansion.
```bash
python3 -m scripts.data.matching_repobench_graphs -search_policy "knn-pattern" --rsg_path "YOUR RSG PATH" --output data/repobench/repos_graphs_labeled 
```

### Training
We can train GNN linker seperately using following script

```bash
CUDA_VISIBLE_DEVICES=0 deepspeed train_gnn.py --deepspeed --deepspeed_config ds_config.json --arch GraphSage --layers 1 --data-path data/repobench/repos_graphs_labeled_cosine_radius_unix --output data/repobench/gnn_model --num-epochs 10 --batch-size 16
```

### Evaluation for RepoBench-P

We can evaluate the model using the following script

```bash
python3 scripts/evaluate_llm.py --data data/repobench/repos_graphs_matched_retrieved --model "gpt3.5" --num-workers 8
```
