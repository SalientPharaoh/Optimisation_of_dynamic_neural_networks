import sys
import os
import shutil
from copy import deepcopy
import deepspeed
import torch
import torch.nn.utils.prune as prune
import torch.optim as optim
import wandb
from transformers import Trainer, TrainingArguments
import numpy as np
import torch.nn as nn


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from models.model_handler import ModelHandler
from data.dataset_loader import DatasetLoader
from models.model_finetuner import ModelFinetuner
from getSysStat import evaluate_model
from optimisers.compression import *
from optimisers.quantizations import *
from optimisers.early_exit import *

def deepspeed_optimisers(model, batch_size=1, group_size=4, quant_type="symmetric",quantization_bits=4):
    ds_config = {
        "train_batch_size": batch_size,
        "gradient_accumulation_steps": batch_size,
        "train_micro_batch_size_per_gpu": batch_size,
        "fp16": {
            "enabled": False
        },
        "optimizer": {"type": "AdamW"},
        "zero_quant": {
            "enabled": True,
            "group_size": group_size
        },
        "compression": {
            "weight_quantization": {
                "shared_groups": True,
                "group_size": group_size,
                "quant_type": quant_type
            },
            "activation_quantization": {
                "enabled": True,
                "quant_type": quant_type,
                "quantization_bits": quantization_bits
            },
            "extreme_compression": {
                "enabled": True,
                "quantization_bits": quantization_bits
            }
        }
    }
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    model_engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config_params=ds_config)
    return model_engine


def unstructured_pruning(model, amount=0.3):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    for module, param in parameters_to_prune:
        prune.remove(module, 'weight')
    return model

    
def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=1)
    accuracy = np.mean(preds == labels)
    return {"accuracy": accuracy}

def base_inference(model_name, dataset_name, wandb_token='647b16aace7d93fde0bdf657d2e927b10fcc8799', hf_token ="", tune = True,subset=None):
    if len(hf_token)!=0:
        from huggingface_hub import login
        login(token=hf_token)

    model_name = model_name
    dataset_name = dataset_name
    subset = subset
    model_handler = ModelHandler(model_name)
    model_handler.load_model()

    dataset_loader = DatasetLoader(model_handler.tokenizer, dataset_name,subset=subset)
    if dataset_name == "stanfordnlp/sst2":
        dataset_loader.prepare_sst2()
    
    training_args = TrainingArguments(
        output_dir='./Baseline',
        evaluation_strategy='epoch',
        logging_dir='./logs',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    )
    trainer = Trainer(
        model=model_handler.model,
        args=training_args,
        train_dataset=dataset_loader.train_dataset,
        eval_dataset=dataset_loader.test_dataset,
        data_collator=dataset_loader.data_collator,
        compute_metrics=compute_metrics
    )
    
    if tune:
        trainer.train()

    eval_results = evaluate_model(trainer)
    return eval_results

def opt_inference(model_name, dataset_name, wandb_token='647b16aace7d93fde0bdf657d2e927b10fcc8799', hf_token ="", tune = True,subset=None):

    if len(hf_token)!=0:
        from huggingface_hub import login
        login(token=hf_token)

    model_name = model_name
    dataset_name = dataset_name
    subset = subset
    model_handler = ModelHandler(model_name)
    model_handler.load_model()

    dataset_loader = DatasetLoader(model_handler.tokenizer, dataset_name,subset=subset)
    if dataset_name == "stanfordnlp/sst2":
        dataset_loader.prepare_sst2()
    
    model = deepcopy(model_handler.model)
    model = unstructured_pruning(model, amount=0.4)
    prepared_model = deepspeed_optimisers(model)

    training_args = TrainingArguments(
        output_dir='./Optimised',
        evaluation_strategy='epoch',
        logging_dir='./logs',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    )
    trainer = Trainer(
        model=prepared_model.module,
        args=training_args,
        train_dataset=dataset_loader.train_dataset,
        eval_dataset=dataset_loader.test_dataset,
        data_collator=dataset_loader.data_collator,
        compute_metrics=compute_metrics
    )
   
    if tune:
        trainer.train()

    eval_results = evaluate_model(trainer)

    output_dir = './OptimisedModel'
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    shutil.make_archive(output_dir, 'zip', output_dir)
    return eval_results

def zeroQuant(model_name, dataset_name, wandb_token='647b16aace7d93fde0bdf657d2e927b10fcc8799', hf_token ="", tune = True,subset=None):

    if len(hf_token)!=0:
        from huggingface_hub import login
        login(token=hf_token)

    model_name = model_name
    dataset_name = dataset_name
    subset = subset
    model_handler = ModelHandler(model_name)
    model_handler.load_model()

    dataset_loader = DatasetLoader(model_handler.tokenizer, dataset_name,subset=subset)
    if dataset_name == "stanfordnlp/sst2":
        dataset_loader.prepare_sst2()
    
    model = deepcopy(model_handler.model)
    prepared_model = zero_quant(model)

    training_args = TrainingArguments(
        output_dir='./Optimised',
        evaluation_strategy='epoch',
        logging_dir='./logs',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    )
    trainer = Trainer(
        model=prepared_model.module,
        args=training_args,
        train_dataset=dataset_loader.train_dataset,
        eval_dataset=dataset_loader.test_dataset,
        data_collator=dataset_loader.data_collator,
        compute_metrics=compute_metrics
    )
   
    if tune:
        trainer.train()

    eval_results = evaluate_model(trainer)
    return eval_results


def XTC(model_name, dataset_name, wandb_token='647b16aace7d93fde0bdf657d2e927b10fcc8799', hf_token ="", tune = True,subset=None):

    if len(hf_token)!=0:
        from huggingface_hub import login
        login(token=hf_token)

    model_name = model_name
    dataset_name = dataset_name
    subset = subset
    model_handler = ModelHandler(model_name)
    model_handler.load_model()

    dataset_loader = DatasetLoader(model_handler.tokenizer, dataset_name,subset=subset)
    if dataset_name == "stanfordnlp/sst2":
        dataset_loader.prepare_sst2()
    
    model = deepcopy(model_handler.model)
    prepared_model = extreme_compression(model)

    training_args = TrainingArguments(
        output_dir='./Optimised',
        evaluation_strategy='epoch',
        logging_dir='./logs',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    )
    trainer = Trainer(
        model=prepared_model.module,
        args=training_args,
        train_dataset=dataset_loader.train_dataset,
        eval_dataset=dataset_loader.test_dataset,
        data_collator=dataset_loader.data_collator,
        compute_metrics=compute_metrics
    )

   
    if tune:
        trainer.train()

    eval_results = evaluate_model(trainer)
    return eval_results



def weight_Quant(model_name, dataset_name, wandb_token='647b16aace7d93fde0bdf657d2e927b10fcc8799', hf_token ="", tune = True,subset=None):

    if len(hf_token)!=0:
        from huggingface_hub import login
        login(token=hf_token)

    model_name = model_name
    dataset_name = dataset_name
    subset = subset
    model_handler = ModelHandler(model_name)
    model_handler.load_model()

    dataset_loader = DatasetLoader(model_handler.tokenizer, dataset_name,subset=subset)
    if dataset_name == "stanfordnlp/sst2":
        dataset_loader.prepare_sst2()
    
    model = deepcopy(model_handler.model)
    prepared_model = weight_quantization(model)

    training_args = TrainingArguments(
        output_dir='./Optimised',
        evaluation_strategy='epoch',
        logging_dir='./logs',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    )
    trainer = Trainer(
        model=prepared_model.module,
        args=training_args,
        train_dataset=dataset_loader.train_dataset,
        eval_dataset=dataset_loader.test_dataset,
        data_collator=dataset_loader.data_collator,
        compute_metrics=compute_metrics
    )

   
    if tune:
        trainer.train()

    eval_results = evaluate_model(trainer)
    return eval_results



def pruning(model_name, dataset_name, wandb_token='647b16aace7d93fde0bdf657d2e927b10fcc8799', hf_token ="", tune = True,subset=None):

    if len(hf_token)!=0:
        from huggingface_hub import login
        login(token=hf_token)

    model_name = model_name
    dataset_name = dataset_name
    subset = subset
    model_handler = ModelHandler(model_name)
    model_handler.load_model()

    dataset_loader = DatasetLoader(model_handler.tokenizer, dataset_name,subset=subset)
    if dataset_name == "stanfordnlp/sst2":
        dataset_loader.prepare_sst2()
    
    model = deepcopy(model_handler.model)
    prepared_model = unstructured_pruning(model)

    training_args = TrainingArguments(
        output_dir='./Optimised',
        evaluation_strategy='epoch',
        logging_dir='./logs',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    )
    trainer = Trainer(
        model=prepared_model,
        args=training_args,
        train_dataset=dataset_loader.train_dataset,
        eval_dataset=dataset_loader.test_dataset,
        data_collator=dataset_loader.data_collator,
        compute_metrics=compute_metrics
    )

   
    if tune:
        trainer.train()

    eval_results = evaluate_model(trainer)
    return eval_results


def PTQuant(model_name, dataset_name, wandb_token='647b16aace7d93fde0bdf657d2e927b10fcc8799', hf_token ="", tune = True,subset=None):

    if len(hf_token)!=0:
        from huggingface_hub import login
        login(token=hf_token)

    model_name = model_name
    dataset_name = dataset_name
    subset = subset
    model_handler = ModelHandler(model_name)
    model_handler.load_model()

    dataset_loader = DatasetLoader(model_handler.tokenizer, dataset_name,subset=subset)
    if dataset_name == "stanfordnlp/sst2":
        dataset_loader.prepare_sst2()
    
    model = deepcopy(model_handler.model)
    prepared_model = PT_Quant(model)

    training_args = TrainingArguments(
        output_dir='./Optimised',
        evaluation_strategy='epoch',
        logging_dir='./logs',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    )
    trainer = Trainer(
        model=prepared_model,
        args=training_args,
        train_dataset=dataset_loader.train_dataset,
        eval_dataset=dataset_loader.test_dataset,
        data_collator=dataset_loader.data_collator,
        compute_metrics=compute_metrics
    )

   
    if tune:
        trainer.train()

    eval_results = evaluate_model(trainer)
    return eval_results


def w8a8_Quant(model_name, dataset_name, wandb_token='647b16aace7d93fde0bdf657d2e927b10fcc8799', hf_token ="", tune = True,subset=None):

    if len(hf_token)!=0:
        from huggingface_hub import login
        login(token=hf_token)

    model_name = model_name
    dataset_name = dataset_name
    subset = subset
    model_handler = ModelHandler(model_name)
    model_handler.load_model()

    dataset_loader = DatasetLoader(model_handler.tokenizer, dataset_name,subset=subset)
    if dataset_name == "stanfordnlp/sst2":
        dataset_loader.prepare_sst2()
    
    model = deepcopy(model_handler.model)
    prepared_model = w8a8_quantization(model)

    training_args = TrainingArguments(
        output_dir='./Optimised',
        evaluation_strategy='epoch',
        logging_dir='./logs',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    )
    trainer = Trainer(
        model=prepared_model.module,
        args=training_args,
        train_dataset=dataset_loader.train_dataset,
        eval_dataset=dataset_loader.test_dataset,
        data_collator=dataset_loader.data_collator,
        compute_metrics=compute_metrics
    )
   
    if tune:
        trainer.train()

    eval_results = evaluate_model(trainer)
    return eval_results