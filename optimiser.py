import deepspeed
import torch
import torch.nn.utils.prune as prune
import torch.optim as optim
from flask import Flask, request, jsonify
import wandb
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset
import numpy as np

def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=1)
    accuracy = np.mean(preds == labels)
    return {"accuracy": accuracy}

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


model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
dataset_name = "stanfordnlp/sst2"
huggingface_token = ""
wandb_token = '647b16aace7d93fde0bdf657d2e927b10fcc8799'
model_subset = 'sequence-classification'
wandb.login(key=wandb_token)

run = wandb.init(project="D2NN-flask", name="baseline")
run_id = run.id
print(run_id)

if len(huggingface_token)!=0:
    from huggingface_hub import login
    login(token=huggingface_token)

dataset = load_dataset(dataset_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

pruned = unstructured_pruning(model, amount=0.3)
model_engine = deepspeed_optimisers(pruned, batch_size=1, group_size=4, quant_type="symmetric",quantization_bits=4)
model = model_engine.module

def preprocess_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding=True, max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(10000)) 
test_dataset = tokenized_datasets['validation'].shuffle(seed=42)
data_collator = DataCollatorWithPadding(tokenizer, return_tensors='pt')

training_args = TrainingArguments(
    output_dir='./Baseline_DistilBERT',
    evaluation_strategy='epoch',
    logging_dir='./logs',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
eval_results = trainer.evaluate()
wandb.log(eval_results)
print(eval_results, run_id)
