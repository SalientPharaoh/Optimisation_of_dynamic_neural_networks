import deepspeed
import torch.optim as optim

def extreme_compression(model, batch_size=1, quantization_bits=8):
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "compression": {
            "extreme_compression": {
                "enabled": True,
                "quantization_bits": quantization_bits
            }
        }
    }
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    model_engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config_params=ds_config)
    return model_engine


def zero_quant(model, batch_size=1, group_size=4):
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "zero_quant": {
            "enabled": True,
            "group_size": group_size
        }
    }
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    model_engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config_params=ds_config)
    return model_engine

