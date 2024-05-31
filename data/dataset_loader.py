import torch
from datasets import load_dataset

class DatasetLoader:
    def __init__(self, tokenizer, dataset_name,subset=None):
        self.dataset_name = dataset_name
        self.subset = subset
        self.dataset = self.load()
        self.tokenizer = tokenizer

    def load(self):
        if(self.subset):
            return load_dataset(self.dataset_name, self.subset, split='test')
        else:
            return load_dataset(self.dataset_name, split='test')

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