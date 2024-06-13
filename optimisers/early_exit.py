import torch
import torch.nn as nn
import time
import torch.optim as optim


class DynamicEarlyExitModel(nn.Module):
    def __init__(self, model, num_classes=2, exit_threshold=0.8, num_exits=3):
        super(DynamicEarlyExitModel, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.exit_threshold = exit_threshold
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.exit_layers = self.determine_exit_layers(self.num_hidden_layers, num_exits)
        self.classifiers = nn.ModuleList([nn.Linear(self.model.config.hidden_size, num_classes) for _ in self.exit_layers])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.total_time = 0

    def determine_exit_layers(self, total_layers, num_exits):
        return [total_layers // (num_exits + 1) * (i + 1) for i in range(num_exits)]

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        for i, layer_index in enumerate(self.exit_layers):
            layer_output = hidden_states[layer_index][:, 0, :]
            logits = self.classifiers[i](layer_output)
            confidence, predicted = torch.max(torch.softmax(logits, dim=1), 1)
            if confidence.item() >= self.exit_threshold:
                return logits

        final_output = hidden_states[-1][:, 0, :]
        final_logits = self.classifiers[-1](final_output)
        return final_logits