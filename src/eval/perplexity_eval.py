import os
import sys
import csv
import glob
import math
import logging
from tqdm import tqdm
from typing import List
import datasets
from datasets import load_dataset, Dataset
from dataclasses import dataclass, field

from knnlm import KNNWrapper, KNNSaver, KEY_TYPE, DIST
import transformers
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from transformers.testing_utils import CaptureLogger

logger = logging.getLogger(__name__)

def repo_to_csv(repo_path) -> str:
    all_file_texts = []
    extensions = ['*.java', '*.c', '*.cpp', '*.cc', '*.cuda', '*.js', '*.go', '*.py', "*jsx"]
    # Walk through each directory in the repo
    for dirpath, dirnames, filenames in os.walk(repo_path):
        # Find all files with the specified extensions in the current directory
        for extension in extensions:
            for filename in glob.glob(os.path.join(dirpath, extension)):
                # Open each file and append its contents to the list of texts
                with open(filename, 'r', encoding='utf-8', errors='ignore') as infile:
                    all_file_texts.append(infile.read())
    if not os.path.exists("data/perplexity/dump_repo_csv"):
        os.mkdir("data/perplexity/dump_repo_csv")
    csv_path = os.path.join("data/perplexity/dump_repo_csv", f"{os.path.basename(repo_path)}.csv")
    total_text = 0
    with open(csv_path, 'w+', newline='') as csvfile:
        fieldnames = ['text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for text in all_file_texts:
            writer.writerow({'text': text})
            total_text += len(text)
    return csv_path, total_text

@dataclass
class KNNArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    knn: bool = field(default=False)
    knn_gpu: bool = field(default=True)
    dstore_size: int = field(default=None, metadata={"help": "The size of the dstore."})
    knn_keytype: KEY_TYPE.from_string = field(default=KEY_TYPE.last_ffn_input)
    save_knnlm_dstore: bool = field(default=False)
    dstore_dir: str = field(default="checkpoints/datastore/")
    knn_sim_func: DIST.from_string = field(default=DIST.l2)
    lmbda: float = field(default=0.25)
    k: int = field(default=1024)
    knn_temp: float = field(default=1.0)
    # Args for building the faiss index:
    build_index: bool = field(default=False)
    # faiss_index: str = field(default="checkpoints/index")
    ncentroids: int = field(default=4096)
    code_size: int = field(default=64)
    probe: int = field(default=32)
    num_keys_to_add_at_a_time: int = field(default=1000000)
    move_dstore_to_mem: bool = field(default=True)
    no_load_keys: bool = field(default=True)
    recompute_dists: bool = field(default=False)

    ## RetoMaton args:
    retomaton: bool = field(default=False)
    cluster_dstore: bool = field(default=False)
    no_pointer: bool = field(default=False)
    min_knns: int = field(default=1)
    max_knns: int = field(default=1024)
    num_clusters: int = field(default=500000)
    sample_size: int = field(default=20000000)
    members: str = field(default=None)
    
class KNNEvaluator:
    """ 2 passes through model, 
    first pass is using self.train through model + saver -> output -> create index
    second pass is using self.val through model + knn (initialize with the index) -> output -> eval
    """
    def __init__(self, 
                 model, 
                 tokenizer, 
                 dstore_dir="checkpoints/datastore/",
                 device='cuda:0', 
                 batch_size=8):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.tokenizer = tokenizer
        
        self.batch_size = batch_size
        self.max_train_examples = 1000
        self.max_eval_examples = 1000
        # self.max_test_examples = 1000
        self.block_size = None # ! params
        self.stride = 512 # ! params
        self.padding_index = -100 # ! params
        self.patience = None
        
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None
        self.raw_datasets = None
        self.output_dir = "checkpoints/" # ! param to save model after train, but in this version, we do not train anything, so this is just a dummy dump folder
        self.training_args = TrainingArguments(self.output_dir, per_device_train_batch_size=self.batch_size, per_device_eval_batch_size=self.batch_size)
        
        # * args for knn saver and knn
        self.dstore_size = None # ! params
        self.dstore_dir = dstore_dir # ! params
        
        # * Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        log_level = self.training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, n_gpu: {self.training_args.n_gpu}"
            + f"distributed training: {bool(self.training_args.local_rank != -1)}, 16-bits training: {self.training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {self.training_args}")
        
    def dataset_setup(self, args):
        if args.type == "transformer":
            raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)

            # Create a new subset of the dataset containing only 10% of the examples
            raw_datasets["train"] = raw_datasets["train"].select(range(len(raw_datasets["train"]) // 500))
            raw_datasets["validation"] = raw_datasets["validation"].select(range(len(raw_datasets["validation"]) // 500))
            
            if "test" in raw_datasets.keys():
                del raw_datasets["test"]
            print(raw_datasets["validation"])
            
            if "validation" not in raw_datasets.keys():
                # raw_datasets["test"] = load_dataset(
                #     self.dataset_name,
                #     self.dataset_config_name,
                #     split=f"train[:{10}%]",
                # )
                raw_datasets["validation"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[:{10}%]",
                )
                raw_datasets["train"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[{10}%:]",
                )
            self.raw_datasets = raw_datasets
        elif args.type == "text_repo": 
            logger.info(f"^^^^^^^Looking into repo path: {args.repo_path} ^^^^^^^^^^^^")
            csv_path, total_text = repo_to_csv(args.repo_path)
            logger.info(f"Done extracting repo to csv: {csv_path}")
            if total_text < 25000:
                raise ValueError(f"Repo {args.repo_path} has less than 25000 characters, please check")
            # from a list of texts to a dataset, with single column named text
            text_dataset = Dataset.from_csv(csv_path)
            train_size = 0.8  # Proportion of the dataset for training
            val_size = 0.2   # Proportion of the dataset for validation
            self.raw_datasets = text_dataset.train_test_split(
                train_size=train_size,
                test_size=val_size,
                shuffle=True
            )
            self.raw_datasets["validation"] = self.raw_datasets["test"]
            del self.raw_datasets["test"]
            logger.info(f"Repo stat: {self.raw_datasets}")
            
    def dataloader_setup(self):
        if self.raw_datasets is None:
            raise ValueError("Please call dataset_setup() first")
        # preprocess the dataset
        column_names = self.raw_datasets["train"].column_names
        text_column_name = column_names[0]
        
        tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

        def tokenize_function(examples):
            with CaptureLogger(tok_logger) as cl:
                instances = []
                for instance in examples[text_column_name]:
                    if instance is not None:
                        instances.append(instance) 
                output = self.tokenizer(instances)
            # clm input could be much much longer than block_size
            if "Token indices sequence length is longer than the" in cl.out:
                tok_logger.warning(
                    "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                )
            return output
        
        with self.training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = self.raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=1, # ! param
                remove_columns=column_names,
                desc="Running tokenizer on dataset",
            )
        
        if self.block_size is None:
            self.block_size = self.tokenizer.model_max_length
            if self.block_size > 1024:
                print("The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value by passing --block_size xxx.")
                self.block_size = 1024
        else:
            if self.block_size > self.tokenizer.model_max_length:
                print(f"The block_size passed ({self.block_size}) is larger than the maximum length for the model"
                    f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
                )
            self.block_size = min(self.block_size, self.tokenizer.model_max_length)
        
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= self.block_size:
                total_length = (total_length // self.block_size) * self.block_size

            input_ids = []
            attention_mask = []
            labels = []
            # We implement a sliding window, so all tokens have a non-zero context in their prediction.
            # We then mask the duplicate tokens' labels, to not count any token twice in the loss.
            for i in tqdm(range(0, total_length, self.stride), total=total_length):
                begin_loc = max(i + self.stride - self.block_size, 0)
                end_loc = min(i + self.stride, total_length)
                trg_len = end_loc - i
                cur_input_ids = concatenated_examples['input_ids'][begin_loc:end_loc]
                cur_labels = list(cur_input_ids)
                cur_labels[:-trg_len] = [self.padding_index] * (len(cur_labels) - trg_len)

                if len(cur_input_ids) < self.block_size:
                    padding_size = self.block_size - len(cur_input_ids)
                    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                    cur_input_ids += [pad_token_id] * padding_size
                    cur_labels += [self.padding_index] * padding_size
                
                input_ids.append(cur_input_ids)
                attention_mask.append([1] * len(cur_labels))
                labels.append(cur_labels)

            result = {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}
            return result

        with self.training_args.main_process_first(desc="grouping texts together"):
            lm_datasets = tokenized_datasets.map(
                group_texts, 
                batched=True,
                num_proc=1, # !param
                desc=f"Grouping texts in chunks of {self.block_size}"
            )
        
        for split, data in lm_datasets.items():
            total_eval_tokens = 0       
            print(split)
            for chunk in data['labels']:
                total_eval_tokens += len([x for x in chunk[1:] if x != self.padding_index])
            print(f'[{split}] Total eval tokens: {total_eval_tokens}')
            if self.dstore_size is None and split == 'train':
                self.dstore_size = total_eval_tokens

        # * debugging
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        self.train_dataset = lm_datasets["train"]
        # if self.max_train_examples is not None:
        #     self.train_dataset = self.train_dataset.select(range(self.max_train_examples))

        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        self.eval_dataset = lm_datasets["validation"]
        # if self.max_eval_examples is not None:
        #     self.eval_dataset = self.eval_dataset.select(range(self.max_eval_examples))
        
        # self.test_dataset = lm_datasets["test"]
        # if self.max_test_examples is not None:
        #     self.test_dataset = self.test_dataset.select(range(self.max_test_samples))
        if self.train_dataset is None or self.eval_dataset is None:
            raise ValueError
        if self.dstore_size is None:
            logger.error("dstore_size is None, please set it up or the self.dataset_setup() function will do it for you, but it fail somehow")
            raise ValueError
        logger.info("dataset setup done")
        
        self.knn_args = KNNArguments(dstore_size=self.dstore_size, dstore_dir=self.dstore_dir)
        
    def run_eval_save_datastore(self):
        """ We are building the datastore for self.train_dataset, but we do not train anything here
        """
        self.saver = KNNSaver(
            dstore_size=self.knn_args.dstore_size, 
            dstore_dir=self.knn_args.dstore_dir, 
            dimension=self.model.config.hidden_size, 
            knn_keytype=self.knn_args.knn_keytype, 
            device=self.device)
        self.saver.break_into(self.model)
        
        # during running datastore saver, we will build the index on the fly, run inference on train
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=self.train_dataset, # ! notice
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)] if self.patience is not None else None,
        )
        trainer.evaluate()
        
        metrics = trainer.evaluate()
        max_train_exmaples = self.max_train_examples if self.max_train_examples is not None else len(self.train_dataset)
        metrics["eval_samples"] = min(max_train_exmaples, len(self.train_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        
        knn_metrics = self.saver.get_metrics()
        metrics.update(knn_metrics)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        self.saver.build_index()
        self.saver.break_out()
        return perplexity
    
    def run_eval_knn_perplexity(self):
        self.knn_wrapper = KNNWrapper(
            dstore_size=self.knn_args.dstore_size, 
            dstore_dir=self.knn_args.dstore_dir, 
            dimension= self.model.config.hidden_size, 
            knn_sim_func=self.knn_args.knn_sim_func, 
            knn_keytype=self.knn_args.knn_keytype,
            no_load_keys=self.knn_args.no_load_keys, 
            move_dstore_to_mem=self.knn_args.move_dstore_to_mem, 
            knn_gpu=self.knn_args.knn_gpu,
            recompute_dists=self.knn_args.recompute_dists,
            k=self.knn_args.k, 
            lmbda=self.knn_args.lmbda, 
            knn_temp=self.knn_args.knn_temp, 
            probe=self.knn_args.probe,
            device=self.device)
        self.knn_wrapper.break_into(self.model)
        
        # Based on storage on train, we do eval on self.val_dataset
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=self.eval_dataset, # ! notice
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)] if self.patience is not None else None,
        )
        trainer.evaluate()
        
        metrics = trainer.evaluate()
        max_val_examples = self.max_eval_examples if self.max_eval_examples is not None else len(self.val_datset)
        metrics["eval_samples"] = min(max_val_examples, len(self.eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        
        knn_metrics = self.knn_wrapper.get_metrics()
        metrics.update(knn_metrics)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        self.knn_wrapper.break_out()
        return perplexity
    
    def run_eval_codegen_perplexity(self):
        # Based on storage on train, we do eval on self.val_dataset
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=self.eval_dataset, # ! notice
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.patience)] if self.patience is not None else None,
        )
        trainer.evaluate()
        
        metrics = trainer.evaluate()
        max_val_examples = self.max_eval_examples if self.max_eval_examples is not None else len(self.val_datset)
        metrics["eval_samples"] = min(max_val_examples, len(self.eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        return perplexity