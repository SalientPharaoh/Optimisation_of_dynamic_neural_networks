import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import numpy as np
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from models.model_handler import ModelHandler

class ModelFinetuner:
    def __init__(self, model_name, dataset_name, subset=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.tokenizer = self.load_model(model_name)
        self.dataset = load_dataset(dataset_name, subset) if subset else load_dataset(dataset_name)
        self.train_dataset, self.test_dataset = self.load_data()
        self.data_collator = DataCollatorWithPadding(self.tokenizer, return_tensors='pt')
    
    def load_model(self,model_name):
        handler = ModelHandler(model_name)
        handler.load_model()
        return handler.model.to(self.device), handler.tokenizer
    
    def data_preprocessing(self, examples, max_length=128):
        return self.tokenizer(examples['sentence'], truncation=True, padding=True, max_length=max_length)
    
    def load_data(self):
        tokenized_datasets = self.dataset.map(self.data_preprocessing, batched=True)
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        train_dataset = tokenized_datasets['train'].shuffle(seed=42)
        test_dataset = tokenized_datasets['validation'].shuffle(seed=42)
        return train_dataset, test_dataset

    def fine_tune(self, epochs = 5, train_batch = 8, eval_batch = 8):
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy='epoch',
            logging_dir='./logs',
            num_train_epochs=epochs,
            per_device_train_batch_size=train_batch,
            per_device_eval_batch_size=eval_batch,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )
        trainer.train()
        return trainer.model
    

    def compute_metrics(self,p):
        predictions, labels = p
        preds = np.argmax(predictions, axis=1)
        accuracy = np.mean(preds == labels)
        return {"accuracy": accuracy}
    
    def evaluate(self, batch_size=8):
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy='epoch',
            logging_dir='./logs',
            per_device_eval_batch_size=batch_size,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=self.test_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )
    
        results = trainer.evaluate()
        return results