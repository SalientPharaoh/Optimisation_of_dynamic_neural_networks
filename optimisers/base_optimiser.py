import deepspeed

# Enabling full precision
def fp16_optimisation(model, batch_size=1):
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "fp16": {
            "enabled": True
        }
    }
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    model_engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config_params=ds_config)
    return model_engine


def zero_optimisation(model, batch_size=1, allgather_bucket_size=2e8, reduce_bucket_size=2e8):
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": allgather_bucket_size,
            "reduce_scatter": True,
            "reduce_bucket_size": reduce_bucket_size,
            "overlap_comm": True,
            "contiguous_gradients": True
        }
    }
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    model_engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config_params=ds_config)
    return model_engine