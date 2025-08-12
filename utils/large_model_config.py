#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
large model config and strategy
for handling large LLM models on multiple GPUs
"""

import torch
import os

def get_model_strategy(model_path, available_memory_gb=None):
    
    if available_memory_gb is None:
        available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    num_gpus = torch.cuda.device_count()
    
    estimated_model_size_gb = estimate_model_size(model_path)
    
    strategy = {
        "use_model_parallel": False,
        "use_cpu_offload": False,
        "use_8bit": False,
        "device_map": "auto",
        "max_memory": None,
        "estimated_size_gb": estimated_model_size_gb
    }
    
    print(f"Estimated model size: {estimated_model_size_gb:.1f}GB")
    print(f"Available memory per GPU: {available_memory_gb:.1f}GB")
    print(f"Number of GPUs: {num_gpus}")
    
    if estimated_model_size_gb > available_memory_gb:
        if num_gpus > 1:
            print("Using model parallelism across multiple GPUs")
            strategy["use_model_parallel"] = True
            strategy["device_map"] = "auto"
            strategy["max_memory"] = {i: f"{int(available_memory_gb * 0.85)}GB" for i in range(num_gpus)}
        else:
            print("Model too large for single GPU, enabling optimizations")
            strategy["use_cpu_offload"] = True
            strategy["use_8bit"] = True
            strategy["device_map"] = "auto"
            strategy["max_memory"] = {0: f"{int(available_memory_gb * 0.85)}GB", "cpu": "50GB"}
    else:
        print("Model fits in single GPU")
        strategy["device_map"] = {"": 0}
    
    return strategy

def estimate_model_size(model_path):

    model_name = model_path.lower()

    if "72b" in model_name or "70b" in model_name:
        return 140  # about 140GB (fp16)
    elif "34b" in model_name or "33b" in model_name:
        return 68   # about 68GB
    elif "14b" in model_name or "13b" in model_name:
        return 28   # about 28GB
    elif "8b" in model_name or "7b" in model_name:
        return 16   # about 16GB
    elif "3b" in model_name:
        return 6    # about 6GB
    elif "1.5b" in model_name:
        return 3    # about 3GB
    elif "0.5b" in model_name:
        return 1    # about 1GB
    else:
        return 20

def apply_memory_optimizations():

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Applied memory optimizations")

def get_training_config_for_large_model(model_size_gb, num_gpus):

    config = {}
    
    if model_size_gb > 50:  # large model
        config["batch_size_per_gpu"] = 1
        config["gradient_accumulation_steps"] = 8
        config["use_gradient_checkpointing"] = True
        config["use_cpu_optimizer"] = True
    elif model_size_gb > 20:  # large model
        config["batch_size_per_gpu"] = 1
        config["gradient_accumulation_steps"] = 4
        config["use_gradient_checkpointing"] = True
        config["use_cpu_optimizer"] = False
    else:  # medium model
        config["batch_size_per_gpu"] = 2 if num_gpus > 1 else 1
        config["gradient_accumulation_steps"] = 2
        config["use_gradient_checkpointing"] = False
        config["use_cpu_optimizer"] = False
    
    return config

if __name__ == "__main__":
    # test config
    strategy = get_model_strategy("../LLMs/Qwen/Qwen2___5-14B")
    print("Recommended strategy:")
    for key, value in strategy.items():
        print(f"  {key}: {value}")
