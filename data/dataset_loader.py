import torch
from datasets import load_dataset
from transformers import DataCollatorWithPadding

class DatasetLoader:
    def __init__(self, tokenizer, dataset_name,subset=None):
        self.dataset_name = dataset_name
        self.subset = subset
        self.dataset = self.load()
        self.tokenizer = tokenizer
        self.data_collator = None
        self.train_dataset = None
        self.test_dataset = None

    def load(self):
        if(self.subset):
            return load_dataset(self.dataset_name, self.subset)
        else:
            return load_dataset(self.dataset_name)
    
    def prepare_sst2(self):
        def preprocess_function(examples):
            return self.tokenizer(examples['sentence'], truncation=True, padding=True, max_length=128)

        tokenized_datasets = self.dataset.map(preprocess_function, batched=True)
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        self.train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(10000)) 
        self.test_dataset = tokenized_datasets['validation'].shuffle(seed=42)
        self.data_collator = DataCollatorWithPadding(self.tokenizer, return_tensors='pt')


    def prepare_data(self):
        tokenized_data = self.dataset.map(lambda x: self.tokenizer(x['sentence1'], x['sentence2'], truncation=True, padding=True), batched=True)
        tokenized_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        tokenized_data = tokenized_data.shuffle(seed=42)
        return tokenized_data

    def custom_collate_fn(batch):
        input_ids = torch.nn.utils.rnn.pad_sequence([item['input_ids'] for item in batch], batch_first=True)
        attention_mask = torch.nn.utils.rnn.pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
        labels = torch.tensor([item['label'] for item in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    
    def get_dataloader(self, batch_size=8):
        dataset = self.prepare_data()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=self.custom_collate_fn)
        return dataloader