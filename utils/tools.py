# !/usr/bin/env python
# -*- coding: UTF-8 -*-


import os
import json
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def clear_content(args):

    log_dir = f"./log/loss/{args.data}"
    os.makedirs(log_dir, exist_ok=True)
    
    loss_file = f"{log_dir}/loss_singleGPU-{args.num_epochs}epoch.txt"
    with open(loss_file, 'w', encoding='utf-8') as f:
        f.write("")


def loss_statistic(args):

    log_dir = f"./log/loss/{args.data}"
    loss_file = f"{log_dir}/loss_singleGPU-{args.num_epochs}epoch.txt"
    
    if os.path.exists(loss_file):
        with open(loss_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        losses = []
        for line in lines:
            if 'loss:' in line:
                try:
                    loss_value = float(line.split('loss:')[1].strip())
                    losses.append(loss_value)
                except:
                    continue
        
        if losses:
            avg_loss = sum(losses) / len(losses)
            min_loss = min(losses)
            max_loss = max(losses)
            
            print(f"Loss Statistics:")
            print(f"Average Loss: {avg_loss:.6f}")
            print(f"Minimum Loss: {min_loss:.6f}")
            print(f"Maximum Loss: {max_loss:.6f}")
            print(f"Total Steps: {len(losses)}")
        else:
            print("No loss values found in log file.")
    else:
        print(f"Loss log file not found: {loss_file}")


def load_pretrain_embeddings(ent_emb_path, rel_emb_path):

    ent_embs = np.load(ent_emb_path)
    ent_embs = torch.tensor(ent_embs).to(device)
    ent_embs.requires_grad = False

    rel_embs = np.load(rel_emb_path)
    rel_embs = torch.tensor(rel_embs).to(device)
    rel_embs.requires_grad = False

    ent_dim = ent_embs.shape[1]
    rel_dim = rel_embs.shape[1]

    print(ent_dim, rel_dim)

    if ent_dim != rel_dim:  # RotatE
        rel_embs = torch.cat((rel_embs, rel_embs), dim=-1)

    return ent_embs, rel_embs 


def set_random_seed(seed=42):

    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass
    
    print(f"Random seed set to {seed}") 
