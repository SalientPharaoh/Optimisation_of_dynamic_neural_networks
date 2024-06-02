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


def asym_fp_quantization(model):
    def afpq_quantization(tensor, bits=4):
        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1
        pos_values = tensor[tensor > 0]
        neg_values = tensor[tensor < 0]
        pos_scale = pos_values.max().item() / qmax if pos_values.numel() > 0 else 1.0
        neg_scale = neg_values.min().item() / qmin if neg_values.numel() > 0 else 1.0
        pos_tensor = tensor.clamp(min=0) / pos_scale
        neg_tensor = tensor.clamp(max=0) / neg_scale
        quantized_tensor = pos_tensor + neg_tensor
        quantized_tensor = quantized_tensor.round().clamp(qmin, qmax).to(torch.int8)
        
        return quantized_tensor, pos_scale, neg_scale
        
    quantized_model_state_dict = {}
    scales = {}
    for name, param in model.named_parameters():
        quantized_data, pos_scale, neg_scale = afpq_quantization(param.data, bits=4)
        quantized_model_state_dict[name] = quantized_data
        scales[f"{name.replace('.', '_')}_pos_scale"] = torch.tensor(pos_scale, dtype=torch.float32)
        scales[f"{name.replace('.', '_')}_neg_scale"] = torch.tensor(neg_scale, dtype=torch.float32)

    model.load_state_dict(quantized_model_state_dict)
    model.scales = scales
    return model