#!/bin/bash

# Run the Python eval_perplexity_codegen with the specified models and batch sizes
CUDA_VISIBLE_DEVICES=0 python3 -m scripts.experiments.eval_perplexity_knnlm_codegen --model_name "Salesforce/codegen-350M-multi" --batch_size 16
CUDA_VISIBLE_DEVICES=0 python3 -m scripts.experiments.eval_perplexity_knnlm_codegen --model_name "Salesforce/codegen2-1B" --batch_size 8
CUDA_VISIBLE_DEVICES=0 python3 -m scripts.experiments.eval_perplexity_knnlm_codegen --model_name "Salesforce/codegen-2B-multi" --batch_size 8