# !/usr/bin/env python
# -*- coding: UTF-8 -*-


import os
import sys
import time
from typing import List

import fire
import torch
import torch.cuda
import torch.distributed as dist
import transformers
from datasets import load_dataset
from combiner import Knowledge_Combiner

from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.prompter import Prompter
from utils.tools import set_random_seed


class CustomDataCollator:
    """
    custom data collator, process data with embedding_ids
    """
    def __init__(self, tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.padding = padding
        
    def __call__(self, features):

        # separate embedding_ids and query_text, but do not put query_text into batch
        embedding_ids = [feature.pop("embedding_ids") for feature in features]
        query_texts = [feature.pop("query_text", None) for feature in features]
        
        valid_keys = {"input_ids", "attention_mask", "labels"}
        cleaned_features = []
        for feature in features:
            cleaned_feature = {k: v for k, v in feature.items() if k in valid_keys}
            cleaned_features.append(cleaned_feature)
        
        batch = transformers.DataCollatorForSeq2Seq(
            self.tokenizer,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
            padding=self.padding
        )(cleaned_features)
        
        batch["embedding_ids"] = torch.LongTensor(embedding_ids)
        
        # add query_text to batch object's custom attribute 
        # to avoid accelerate processing
        batch._query_texts = query_texts
        
        return batch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomTrainer(transformers.Trainer):

    
    def compute_loss(self, model, inputs, return_outputs=False):

        query_texts = getattr(inputs, '_query_texts', None)
        
        if query_texts is not None:
            inputs["query_text"] = query_texts

        return super().compute_loss(model, inputs, return_outputs)



def train(args):

    if hasattr(args, 'seed'):
        set_random_seed(args.seed)
    
    base_model = args.llm_path
    data_path = args.train_data
    output_dir = args.lora_dir
    ent_embs = args.ent_emb_dir
    rel_embs = args.rel_emb_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # training hyperparams
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    micro_batch_size = args.micro_batch_size
    learning_rate = args.learning_rate
    cutoff_len = args.cutoff_len
    val_set_size = args.val_set_size
    
    # lora hyperparams
    lora_r = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lora_target_modules = args.lora_target_modules
    
    # prefix tuning params
    num_prefix = args.num_prefix
    
    # llm hyperparams
    train_on_inputs = args.train_on_inputs
    add_eos_token = args.add_eos_token
    group_by_length = args.group_by_length
    
    # weight and bias (wandb) params
    wandb_project = args.wandb_project
    wandb_run_name = args.wandb_run_name
    wandb_watch = args.wandb_watch
    wandb_log_model = args.wandb_log_model
    
    # additional params
    resume_from_checkpoint = args.resume_from_checkpoint
    prompt_template_name = args.prompt_template_name
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Qwen-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            # f"cache_dir: {cache_dir}\n"
            f"ent_embs: {ent_embs}\n"
            f"rel_embs: {rel_embs}\n"
            f"num_epochs: {num_epochs}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"num_prefix: {num_prefix}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model'"

    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Training setup - batch_size: {batch_size}, micro_batch_size: {micro_batch_size}")
        print(f"Initial gradient_accumulation_steps: {gradient_accumulation_steps}")
        print(f"World size: {world_size}, DDP enabled: {ddp}")

    if ddp:
        device_map = {"": int(
            os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = max(1, gradient_accumulation_steps * 2)
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"Adjusted gradient_accumulation_steps for DDP: {gradient_accumulation_steps}")
            print(f"Effective batch size: {micro_batch_size * gradient_accumulation_steps * world_size}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
        # cache_dir=cache_dir
    )

    # model = prepare_model_for_int8_training(model)

    # tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    tokenizer.pad_token_id = (
        0
    )
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):

        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):

        instruction = ("As a knowledge graph data analysis expert, you are given a head entity and a relation from a triple of a knowledge graph. "
                      "Your task is to predict the most likely tail entity. Given a list of candidate tail entities, reorder this list so that the most likely "
                      "tail entity is first, followed by the next most likely, and so on. Please only provide the reordered list in the format starting with '[' "
                      "and ending with ']' in your response. Do not provide any explanations or additional text. The candidate tail entities are: {candidates}.")

        full_prompt = prompter.generate_prompt(
            instruction.format(candidates=data_point["Init_list2name"][:args.num_candidates]),
            data_point["Query2name"],
            data_point["Target_list2name"][:args.num_candidates],
        )
        tokenized_full_prompt = tokenize(full_prompt)

        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(instruction.format(candidates=data_point["Init_list2name"][:args.num_candidates]), data_point["Query2name"])  # 用户提示仅包含指令和输入
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]


        query_str = data_point["Query2id"]
        query_parts = query_str.strip("()").split(", ")
        head_id = int(query_parts[0])
        relation_id = int(query_parts[1])
        
        candidate_tail_ids = data_point["Init_list2id"][:args.num_candidates]
        
        embedding_ids = [head_id, relation_id] + candidate_tail_ids
        
        tokenized_full_prompt["embedding_ids"] = embedding_ids
        tokenized_full_prompt["query_text"] = data_point["Query2name"]
        
        return tokenized_full_prompt

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    adapter_config = {
        'adapter_hidden_dim': getattr(args, 'adapter_hidden_dim', None),
        'adapter_num_layers': getattr(args, 'adapter_num_layers', 3),
        'adapter_dropout': getattr(args, 'adapter_dropout', 0.05)
    }
    
    slama_model = Knowledge_Combiner(
        model, 
        num_prefix, 
        args.llm_hidden_size, 
        ent_emb_path=ent_embs, 
        rel_emb_path=rel_embs,
        tokenizer=tokenizer,
        **adapter_config
    )

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:

        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")

        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")
            resume_from_checkpoint = (
                False
            )

        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    # model.print_trainable_parameters()

    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=data["train"].column_names)
        val_data = None
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Training data columns: {train_data.column_names}")
        print(f"Training data size: {len(train_data)}")
        if len(train_data) > 0:
            print(f"Sample data keys: {list(train_data[0].keys())}")

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = CustomTrainer(
        model=slama_model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=False,
            bf16=torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            logging_steps=5,
            optim="adamw_hf",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            save_safetensors=False,
            eval_steps=None,
            save_steps=0.999999,
            output_dir=output_dir,
            save_total_limit=0,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
            run_name=None,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        ),
        data_collator=CustomDataCollator(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict

    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # print(torch.cuda.memory_allocated())
    # print(torch.cuda.memory_reserved())

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        # model.save_pretrained(output_dir)
        model.save_pretrained(output_dir, state_dict=old_state_dict())
        torch.save(slama_model.embeddings, os.path.join(output_dir, "embeddings.pth"))

        print("\n If there's a warning about missing keys above, please disregard.\n")
    
    if ddp:
        dist.barrier()


def calculate_average_loss(file_path, num_epoch):

    with open(file_path, 'r') as file:
        losses = file.readlines()

    loss_values = [float(ls.strip()) for ls in losses]

    total_batches = len(loss_values)
    batches_per_epoch = total_batches // num_epoch

    epoch_losses = []
    for epoch in range(num_epoch):
        start_idx = epoch * batches_per_epoch
        end_idx = (epoch + 1) * batches_per_epoch
        epoch_loss_values = loss_values[start_idx:end_idx]
        average_loss = sum(epoch_loss_values) / len(epoch_loss_values) if epoch_loss_values else 0
        epoch_losses.append(average_loss)

    return epoch_losses


if __name__ == "__main__":

    data = "NELL"
    num_epochs = 20

    loss_dir = f"./log/loss/{data}"
    os.makedirs(loss_dir, exist_ok=True)
    
    if torch.cuda.device_count() == 1:
        with open(f"{loss_dir}/loss_singleGPU-{num_epochs}epoch.txt", "w") as f1:
            pass
    elif torch.cuda.device_count() > 1:
        with open(f"{loss_dir}/loss_multipleGPU-{num_epochs}epoch.txt", "w") as f2:
            pass

    start_time = time.time()
    fire.Fire(train)
    end_time = time.time()
    total_time = (end_time - start_time) / 3600
    print(f"Total time taken: {format(total_time, '.2f')} hours\n")

    if torch.cuda.device_count() == 1:
        epoch_loss = calculate_average_loss(
            file_path=f'{loss_dir}/loss_singleGPU-{num_epochs}epoch.txt',
            num_epoch=num_epochs)
        print(f"Average loss per epoch in single GPU:\n {epoch_loss} ")
    elif torch.cuda.device_count() > 1:
        epoch_loss = calculate_average_loss(
            file_path=f'{loss_dir}/loss_multipleGPU-{num_epochs}epoch.txt',
            num_epoch=num_epochs)
        print(f"Average loss per epoch in multiple GPU:\n: {epoch_loss}")
