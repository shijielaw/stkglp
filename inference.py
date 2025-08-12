# !/usr/bin/env python
# -*- coding: UTF-8 -*-


import re
import json
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.utils import logging
logging.get_logger("transformers").setLevel(logging.ERROR)

import os
from utils.tools import set_random_seed
from utils.large_model_config import get_model_strategy, apply_memory_optimizations


def load_test_dataset(path):
    return json.load(open(path, "r"))


def response_format(text, index, k):
    match = re.search(r'\[([^]]+)]', text)
    sub_match = re.search(r'\[([^]]*)', text)

    def strip_quotes(element):
        element = element.strip()
        element = re.sub(r"^[\"']+|[\"']+$", "", element)
        return element

    if match:
        list_content = match.group(1)
        elements = list_content.split(',')

        final_elements = [strip_quotes(element) for element in elements]

        if len(final_elements) > k:
            final_elements = final_elements[:k]

    elif sub_match:
        list_content = sub_match.group(1)

        elements = list_content.split(',')

        final_elements = [strip_quotes(element) for element in elements]

        if len(final_elements) > k:
            final_elements = final_elements[:k]

    else:
        print(f"No valid list found in the response {index}.\n")
        final_elements = "XXX-XXX"

    return final_elements


def test(args):

    apply_memory_optimizations()
    
    if hasattr(args, 'seed'):
        set_random_seed(args.seed)
    
    max_new_token = args.max_new_tokens
    base_path = args.llm_path
    lora_weights = args.lora_dir
    embedding_path = "{}/embeddings.pth".format(lora_weights)

    test_data_path = args.test_data
    test_dataset = load_test_dataset(test_data_path)

    result_pth = args.result_path
    response_pth = args.response_path
    
    os.makedirs(os.path.dirname(result_pth), exist_ok=True)
    os.makedirs(os.path.dirname(response_pth), exist_ok=True)

    kg_embeddings = torch.load(embedding_path)

    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(base_path, padding_side="left")

    print(f'Loading base model from {base_path}...')
    
    strategy = get_model_strategy(base_path)
    available_gpus = torch.cuda.device_count()
    
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "device_map": strategy["device_map"]
    }
    
    if strategy["max_memory"]:
        model_kwargs["max_memory"] = strategy["max_memory"]
    
    if strategy["use_8bit"]:
        model_kwargs["load_in_8bit"] = True
        print("Using 8-bit quantization")
    
    model = AutoModelForCausalLM.from_pretrained(base_path, **model_kwargs)
    
    if strategy["use_model_parallel"] and available_gpus > 1:
        first_device = list(model.hf_device_map.values())[0]
        cuda = f"cuda:{first_device}" if isinstance(first_device, int) else "cuda:0"
        print(f"Model distributed across GPUs, primary device: {cuda}")
    else:
        cuda = "cuda:0"

    print('Loading LoRA Fine-tuned Model...')
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.bfloat16,
    )
    
    if not strategy["use_model_parallel"]:
        model = model.to(cuda)
    
    kg_embeddings = kg_embeddings.to(cuda)
    print(f"Model and embeddings loaded, primary device: {cuda}")
    
    if strategy["use_model_parallel"] and hasattr(model, 'hf_device_map'):
        print(f"Model device map: {dict(list(model.hf_device_map.items())[:5])}...")

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 151643  # depends on parameter in model config.json
    model.config.eos_token_id = 151643  # depends on parameter in model config.json

    model = model.eval()

    results = []
    responses = []

    for data in tqdm(test_dataset, desc='Testing'):
        index = data.get("Q_id", len(results))
        query_str = data["Query2id"]
        query_parts = query_str.strip("()").split(", ")
        head_id = int(query_parts[0])
        relation_id = int(query_parts[1])
        candidate_tail_ids = data["Init_list2id"][:args.num_candidates]
        embedding_ids = [head_id, relation_id] + candidate_tail_ids
        
        ids = torch.LongTensor(embedding_ids).reshape(1, -1).to(cuda)
        prefix = kg_embeddings(ids)

        instruction_text = ("As a knowledge graph data analysis expert, you are given a head entity and a relation from a triple of a knowledge graph. "
                           "Your task is to predict the most likely tail entity. Given a list of candidate tail entities, reorder this list so that the most likely "
                           "tail entity is first, followed by the next most likely, and so on. Please only provide the reordered list in the format starting with '[' "
                           "and ending with ']' in your response. Do not provide any explanations or additional text. The candidate tail entities are: {}.")
        
        prompt_template = """### Instruction:
        {}

        ### Input:
        {}

        ### Response:

        """
        prompt = prompt_template.format(instruction_text.format(data["Init_list2name"][:args.num_candidates]), data["Query2name"])

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(cuda)

        token_embeds = model.model.model.embed_tokens(input_ids)
        input_embeds = torch.cat((prefix, token_embeds), dim=1)

        generate_ids = model.generate(
            inputs_embeds=input_embeds.to(dtype=torch.bfloat16),
            max_new_tokens=max_new_token
        )

        context = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = response.replace(context, "").strip()

        saved_response = f'INDEX {index}:\n' + response + '\n\n'
        responses.append(saved_response)

        print(f'\n-------------------------------\nINDEX {index}:\n' + str(response) + '\n-------------------------------\n')

        res = response_format(response, index, args.num_candidates)

        results.append(
            {
                "index": index,
                "candidates": data["Init_list2name"][:args.num_candidates],
                "answer": data["Answer2name"],
                "predict": res
            }
        )

    print('Saving responses...')
    with open(response_pth, 'w', encoding='utf-8') as f2:
        f2.writelines(responses)

    print('Saving results...')
    with open(result_pth, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--data_name', type=str, default=None)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--num_candidates', type=int, default=30)
    parser.add_argument('--llm_path', type=str, default='./llms/Qwen/Qwen2___5-14B')
    parser.add_argument('--lora_dir', type=str, default=None)
    parser.add_argument('--test_data', type=str, default=None)
    parser.add_argument('--result_path', type=str, default=None)
    parser.add_argument('--response_path', type=str, default=None)
    
    args = parser.parse_args()

    # =========parameters to be modified==========
    args.data_name = "UMLS"
    args.num_epochs = 1
    # ===========================================
    
    # set default paths
    if args.lora_dir is None:
        args.lora_dir = f"./lora/{args.data_name}/run_epoch{args.num_epochs}"
    if args.test_data is None:
        args.test_data = f"data/{args.data_name}/test.json"
    if args.result_path is None:
        args.result_path = f"./log/res/{args.data_name}/results_epoch{args.num_epochs}.json"
    if args.response_path is None:
        args.response_path = f"./log/res/{args.data_name}/responses_epoch{args.num_epochs}.txt"
    
    test(args)
