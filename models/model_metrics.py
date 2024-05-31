import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time
import torchinfo

class ModelMetrics:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.total_time = 0


    def run_inference(self):
        self.model.eval()
        all_predictions = []
        all_labels = []

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('../log_dir'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for batch in self.dataloader:
                start_time = time.time()
                with record_function("model_inference"):
                    inputs = {key: batch[key].to(self.device) for key in ['input_ids', 'attention_mask']}
                    labels = batch['labels'].to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy().tolist()
                    all_predictions.extend(predictions)
                    all_labels.extend(labels.cpu().numpy().tolist())
                end_time = time.time()
                self.total_time += (end_time - start_time)
                prof.step()

        self._print_summary(prof)
        return all_predictions, all_labels

    def _print_summary(self, prof):
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(f"Total inference time: {self.total_time:.4f} seconds")
        print(f"Throughput: {len(self.dataloader.dataset) / self.total_time:.2f} samples/second")
        
    def calculate_metrics(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        model_size_mb = sum(p.element_size() * p.numel() for p in self.model.parameters()) / (1024 ** 2)
        throughput = len(self.dataloader.dataset) / self.total_time
        mean_time = self.total_time / len(self.dataloader)
        print(f"Model Parameters: {num_params}")
        print(f"Model Size (MB): {model_size_mb:.2f}")
        print(f"FLOPS: {flops}")
        print(f"MACs: {macs}")
        print(f"Throughput: {throughput:.2f} samples/second")
        print(f"Mean Inference Time: {mean_time:.4f} seconds")
    
    def calculate_accuracy():
        predictions, labels = run_inference(self.model, self.dataloader, self.device)
        metric = load_metric('accuracy')
        accuracy =  metric.compute(predictions=predictions, references=labels)['accuracy']
        print(f"Inference Accuracy: {accuracy:.4f}")

        self.calculate_metrics()