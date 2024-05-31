import torch.nn.utils.prune as prune
import deepspeed

def unstructured_pruning(model, amount=0.3):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    for module, param in parameters_to_prune:
        prune.remove(module, 'weight')
    return model


def deepspeed_pruning(model, batch_size=1, method='magnitude', total_steps = 10000, warmup_steps=1000, target_sparsity = 0.7):
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "pruning": {
            "enabled": True,
            "method": "magnitude",  # "magnitude" or "movement"
            "params": {
                "start_step": 0,
                "end_step": total_steps,
                "frequency": 100
            },
            "target_sparsity": target_sparsity,
            "sparsity_warmup": {
                "type": "linear",
                "steps": warmup_steps
            }
        }
    }
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    model_engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config_params=ds_config)
    return model_engine