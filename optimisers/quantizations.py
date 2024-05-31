import deepspeed
import torch
import torch.optim as optim


def weight_quantization(model, batch_size=1, group_size=32, quant_type="symmetric"):
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "compression": {
            "weight_quantization": {
                "shared_groups": True,
                "group_size": group_size,
                "quant_type": quant_type
            }
        }
    }
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    model_engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config_params=ds_config)
    return model_engine


def activation_quantization(model, batch_size=1, quantization_bits=8, quant_type="symmetric"):
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "compression": {
            "activation_quantization": {
                "enabled": True,
                "quant_type": quant_type,
                "quantization_bits": quantization_bits
            }
        }
    }
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    model_engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config_params=ds_config)
    return model_engine


def PT_Quant(model):
    # Quantization only occurs on CPU 
    device = torch.device('cpu')
    quantized_model = torch.quantization.quantize_dynamic(model.to(device), {torch.nn.Linear}, dtype=torch.float16)
    return quantized_model

