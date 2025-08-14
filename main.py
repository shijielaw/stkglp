# !/usr/bin/env python
# -*- coding: UTF-8 -*-


import time
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from finetune import train
from inference import test
from eval import evaluate
from utils.tools import clear_content, loss_statistic, set_random_seed

import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)

os.environ["WANDB_MODE"] = "disabled"  # comment this line to enable wandb
# os.environ["WANDB_API_KEY"] = ""  # set API key (only set key to enable wandb)
# os.environ["WANDB_MODE"] = "offline"   # offline, "wandb sync /path/to/wandb" to upload wandb after training

def build_arg():
    parser = argparse.ArgumentParser(description='Knowledge Graph Link Prediction with LLM.')

    # dataset params
    parser.add_argument('--data_name', type=str, help='name of dataset')
    parser.add_argument('--train_data', type=str, help='path of training dataset')
    parser.add_argument('--valid_data', type=str, help='path of validation dataset')
    parser.add_argument('--test_data', type=str, help='path of test dataset')
    parser.add_argument('--response_path', type=str, help='path of response file')
    parser.add_argument('--result_path', type=str, help='path of result file')
    parser.add_argument('--evaluation_path', type=str, help='path of evaluation file')
    parser.add_argument('--num_candidates', type=int, default=30)

    # embedding params
    parser.add_argument('--ent_emb_dir', type=str, help='directory of entity embedding')
    parser.add_argument('--rel_emb_dir', type=str, help='directory of relation embedding')

    # model paths
    parser.add_argument('--lora_dir', type=str, help='directory of lora weights')
    parser.add_argument('--llm_path', type=str, help='Path of LLM', default='./llms/Qwen/Qwen2___5-14B')

    # lora params
    parser.add_argument('--lora_rank', type=int, help='rank of lora')
    parser.add_argument('--lora_alpha', type=int, help='alpha of lora')
    parser.add_argument('--lora_dropout', type=float, help='dropout of lora', default=0.05)
    parser.add_argument('--lora_target_modules', help='target modules of lora', default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # llm hyperparams
    parser.add_argument('--llm_hidden_size', type=int, help='hidden size of llm')
    parser.add_argument('--train_on_inputs', help='train on inputs', default=True)
    parser.add_argument('--add_eos_token', help='whether ass eos_token in input and output', default=False)
    parser.add_argument('--group_by_length', help='whether group by length, for speed up', default=False)

    # training hyperparams
    parser.add_argument('--num_epochs', type=int, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, help='batch size for training')
    parser.add_argument('--micro_batch_size', type=int, help='micro batch size for training')
    parser.add_argument('--val_set_size', type=float, help='validation set size', default=0)
    parser.add_argument('--learning_rate', type=float, help='learning rate for training', default=1e-5)
    parser.add_argument('--cutoff_len', type=float, help='cutoff length for training', default=1024)

    # weight and bias (wandb) params
    parser.add_argument('--wandb_project', type=str, help='project name for wandb', default='')
    parser.add_argument('--wandb_run_name', type=str, help='run name for wandb', default='')
    parser.add_argument('--wandb_watch', type=str, help='watch model for wandb', default='')
    parser.add_argument('--wandb_log_model', type=str, help='logger model for wandb', default='')

    # additional params
    parser.add_argument('--num_prefix', type=int, help='number of prefix', default=1)
    parser.add_argument('--resume_from_checkpoint', type=str, help='resume from checkpoint', default=None)
    parser.add_argument('--prompt_template_name', type=str, help='prompt template name')
    parser.add_argument('--run_mode', type=str, help='train or just test')
    parser.add_argument('--max_new_tokens', type=int, help='max new tokens for generation', default=512)
    parser.add_argument('--record_loss', type=bool, help='record loss')
    parser.add_argument('--seed', type=int, help='random seed for reproducibility', default=42)
    
    # MLP adapter params
    parser.add_argument('--adapter_hidden_dim', type=int, help='hidden dimension of MLP adapter, None for auto', default=None)
    parser.add_argument('--adapter_num_layers', type=int, help='number of layers in MLP adapter', default=3)
    parser.add_argument('--adapter_dropout', type=float, help='dropout rate for MLP adapter', default=0.05)
    
    # multi-GPU params
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple GPUs for training', default=False)
    parser.add_argument('--master_port', type=str, default='12355', help='master port for distributed training')


    arg = parser.parse_args()

    return arg


def setup_distributed(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def run_distributed_training(rank, world_size, args):
    setup_distributed(rank, world_size, args.master_port)

    set_random_seed(args.seed + rank)
    
    try:
        if args.run_mode == "train":
            if args.record_loss:
                clear_content(args)
            start_time = time.time()

            train(args)

            end_time = time.time()
            total_time = (end_time - start_time) / 3600
            if rank == 0:
                print(f"Total time taken: {format(total_time, '.2f')} hours\n")

            if args.record_loss:
                loss_statistic(args)

            if dist.is_initialized():
                dist.barrier()
            
            if rank == 0:
                cleanup_distributed()
                
                print("Setting up inference environment...")
                
                test(args)
                evaluate(args.result_path, args.eval_path, args.data_name)
                return

        elif args.run_mode == "test":
            if rank == 0:
                cleanup_distributed()
                print("Setting up inference environment...")
                test(args)
                evaluate(args.result_path, args.eval_path, args.data_name)
                return
        else:
            if rank == 0:
                print("Please input the correct run mode, should be 'train' or 'test'!\n")
    
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    
    args = build_arg()

    set_random_seed(args.seed)

    # =========parameters to be modified==================================
    args.data_name = "UMLS"  # UMLS, NELL, Wiki16K, FB15K-237
    args.num_epochs = 1  # just for test
    args.use_multi_gpu = True
    args.run_mode = "train"  # train or test
        
    if args.use_multi_gpu and torch.cuda.device_count() > 1:
        args.batch_size = args.micro_batch_size = 1
        print(f"Using {torch.cuda.device_count()} GPUs for training")
    else:
        args.batch_size = args.micro_batch_size = 1
        args.use_multi_gpu = False

    args.lora_rank = 32
    args.lora_alpha = args.lora_rank * 2

    args.record_loss = False
    # ====================================================================

    if any(model in args.llm_path for model in ["Meta-Llama-3-8B", "Qwen3-8B-Base", "Mistral-7B-Instruct-v0.3"]):
        args.llm_hidden_size = 4096

    elif any(model in args.llm_path for model in ["Qwen3-0.6B-Base", "Qwen3-0___6B-Base"]):
        args.llm_hidden_size = 1024

    elif any(model in args.llm_path for model in ["Qwen2.5-14B", "Qwen2___5-14B", "Llama-2-13b-ms"]):
        args.llm_hidden_size = 5120

    elif any(model in args.llm_path for model in ["Qwen2-7B", "Qwen2.5-7B", "Qwen2___5-7B"]):
        args.llm_hidden_size = 3584

    elif any(model in args.llm_path for model in ["Qwen2.5-3B", "Qwen2___5-3B"]):
        args.llm_hidden_size = 2048

    elif any(model in args.llm_path for model in ["Qwen2-1.5B", "Qwen2-1___5B"]):
        args.llm_hidden_size = 1536

    elif any(model in args.llm_path for model in ["Qwen2-0.5B", "Qwen2-0___5B"]):
        args.llm_hidden_size = 896

    else:
        assert ("Please input the correct LLM path!\n")

    args.train_data = f"./data/{args.data_name}/train.json"
    args.valid_data = f"./data/{args.data_name}/valid.json"
    args.test_data = f"./data/{args.data_name}/test.json"
    args.ent_emb_dir = f"./data/{args.data_name}/entity_embedding.npy"
    args.rel_emb_dir = f"./data/{args.data_name}/relation_embedding.npy"
    args.lora_dir = f"./lora/{args.data_name}/run_epoch{args.num_epochs}"
    args.prompt_template_name = "prompts"

    args.response_path = f"./log/res/{args.data_name}/responses_epoch{args.num_epochs}.txt"
    args.result_path = f"./log/res/{args.data_name}/results_epoch{args.num_epochs}.json"
    args.eval_path = f"./log/res/{args.data_name}/evaluation_epoch{args.num_epochs}.txt"

    if args.use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Multi-GPU training, using {torch.cuda.device_count()} GPUs")
        world_size = torch.cuda.device_count()
        mp.spawn(run_distributed_training, args=(world_size, args), nprocs=world_size, join=True)

    else:
        print("Single-GPU training")
        if args.run_mode == "train":
            if args.record_loss:
                clear_content(args)
            start_time = time.time()

            train(args)

            end_time = time.time()
            total_time = (end_time - start_time) / 3600
            print(f"Total time taken: {format(total_time, '.2f')} hours\n")

            if args.record_loss:
                loss_statistic(args)

            test(args)
            evaluate(args.result_path, args.eval_path, args.data_name)

        elif args.run_mode == "test":
            test(args)
            evaluate(args.result_path, args.eval_path, args.data_name)

        else:
            assert("Please input the correct run mode, should be 'train' or 'test'!\n") 
