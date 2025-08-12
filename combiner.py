# !/usr/bin/env python
# -*- coding: UTF-8 -*-


import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple

from transformers import AutoModelForCausalLM
from utils.tools import load_pretrain_embeddings


class Knowledge_Combiner(nn.Module):
    def __init__(
            self,
            model: AutoModelForCausalLM,
            num_prefix: int,
            dim_llm: int,
            ent_emb_path: str = None,
            rel_emb_path: str = None,
            pretrain_emb_path: str = None,
            adapter_hidden_dim: int = None,
            adapter_num_layers: int = 2,
            adapter_dropout: float = 0.1,
            tokenizer = None
    ) -> None:
        super(Knowledge_Combiner, self).__init__()
        self.llm_model = model
        self.tokenizer = tokenizer
        ent_embs, rel_embs = load_pretrain_embeddings(ent_emb_path, rel_emb_path)

        if pretrain_emb_path is None:
            print("Adapter Trained From Scratch")
            self.embeddings = PretrainKGEmbedding(
                pretrain_ent_embs=ent_embs,
                pretrain_rel_embs=rel_embs,
                dim_llm=dim_llm,
                num_prefix=num_prefix,
                adapter_hidden_dim=adapter_hidden_dim,  # MLP隐藏层维度
                adapter_num_layers=adapter_num_layers,  # MLP层数
                adapter_dropout=adapter_dropout  # Dropout率
            )
        else:
            print("Adapter Load From {}".format(pretrain_emb_path))
            self.embeddings = torch.load(pretrain_emb_path)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.llm_model, 'gradient_checkpointing_enable'):
            self.llm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        if hasattr(self.llm_model, 'gradient_checkpointing_disable'):
            self.llm_model.gradient_checkpointing_disable()
    
    def save_pretrained(self, save_directory, **kwargs):
        if hasattr(self.llm_model, 'save_pretrained'):
            return self.llm_model.save_pretrained(save_directory, **kwargs)
    
    @property
    def config(self):
        return self.llm_model.config if hasattr(self.llm_model, 'config') else None
    
    def _apply_attention_to_candidates(self, kg_embeds, embedding_ids, query_text, input_ids):
        """
        apply attention to candidate embeddings
        
        Args:
            kg_embeds: original KG embeddings [batch_size, seq_len, hidden_dim]
            embedding_ids: embedding IDs [batch_size, num_entities] 
            query_text: Query2name text
            input_ids: input token IDs (for tokenizer)
            
        Returns:
            weighted_kg_embeds: weighted KG embeddings
        """
        batch_size = kg_embeds.shape[0]
        hidden_dim = kg_embeds.shape[2]

        if embedding_ids.shape[1] != 32:  # 32 = 2 (head + relation) + 30 (candidates)
            return kg_embeds

        head_relation_embeds = kg_embeds[:, :2, :]
        candidate_embeds = kg_embeds[:, 2:, :]
        
        if isinstance(query_text, str):
            query_emb = self._get_query_embedding(query_text, batch_size, hidden_dim, kg_embeds.device)
        elif isinstance(query_text, list) and len(query_text) == batch_size:
            query_embeds = []
            for qt in query_text:
                qe = self._get_query_embedding(qt, 1, hidden_dim, kg_embeds.device)
                query_embeds.append(qe)
            query_emb = torch.cat(query_embeds, dim=0)  # [batch_size, 1, hidden_dim]
        else:
            query_emb = head_relation_embeds.mean(dim=1, keepdim=True)
        
        scores = torch.bmm(query_emb, candidate_embeds.transpose(-2, -1))
        attention_weights = torch.softmax(scores, dim=-1)
        weighted_candidate_embeds = candidate_embeds * attention_weights.transpose(-2, -1)
        weighted_kg_embeds = torch.cat([head_relation_embeds, weighted_candidate_embeds], dim=1)
        
        return weighted_kg_embeds
    
    def _get_query_embedding(self, query_text, batch_size, hidden_dim, device):
        """
        get query embedding from Query2name text
        
        Args:
            query_text: Query2name text
            batch_size: batch size
            hidden_dim: hidden dimension
            device: device
            
        Returns:
            query_emb: Query embedding [batch_size, 1, hidden_dim]
        """
        try:
            if self.tokenizer is not None:
                tokenized = self.tokenizer(
                    query_text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=128
                )
                query_tokens = tokenized.input_ids.to(device)
                
                with torch.no_grad():
                    query_emb = self.llm_model.model.model.embed_tokens(query_tokens)
                    query_emb = query_emb.mean(dim=1, keepdim=True)
                
                return query_emb
            else:
                query_emb = torch.randn(batch_size, 1, hidden_dim, device=device) * 0.1
                return query_emb
            
        except Exception as e:
            print(f"Warning: Failed to get query embedding: {e}")
            return torch.zeros(batch_size, 1, hidden_dim, device=device)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            embedding_ids: torch.LongTensor = None,
            query_text: Optional[str] = None
    ):
        kg_embeds = self.embeddings(embedding_ids)
        if query_text is not None and embedding_ids is not None and embedding_ids.shape[1] > 2:
            kg_embeds = self._apply_attention_to_candidates(kg_embeds, embedding_ids, query_text, input_ids)
        
        batch_size, seq_len, _ = kg_embeds.shape
        token_embeds = self.llm_model.model.model.embed_tokens(input_ids)
        
        if kg_embeds.shape[0] != token_embeds.shape[0]:
            raise ValueError(f"Batch size mismatch: kg_embeds batch={kg_embeds.shape[0]}, token_embeds batch={token_embeds.shape[0]}")
        
        input_embeds = torch.cat((kg_embeds, token_embeds), dim=1)
        prefix_mask = torch.ones((batch_size, seq_len), device=kg_embeds.device)
        prefix_labels = torch.full((batch_size, seq_len), fill_value=-100, dtype=torch.long, device=kg_embeds.device)
        new_attention_mask = torch.cat((prefix_mask, attention_mask), dim=-1)
        new_labels = torch.cat((prefix_labels, labels), dim=-1)

        return self.llm_model(
            input_ids=None,
            attention_mask=new_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embeds.to(torch.bfloat16),
            labels=new_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class PretrainKGEmbedding(nn.Module):
    def __init__(
            self,
            pretrain_ent_embs,
            pretrain_rel_embs,
            dim_llm,
            num_prefix,
            adapter_hidden_dim=None,
            adapter_num_layers=2,
            adapter_dropout=0.1
    ):
        super(PretrainKGEmbedding, self).__init__()
        self.num_prefix = num_prefix
        self.llm_dim = dim_llm
        self.emb_dim = num_prefix * dim_llm
        self.ent_embeddings = nn.Embedding.from_pretrained(pretrain_ent_embs)
        self.rel_embeddings = nn.Embedding.from_pretrained(pretrain_rel_embs)
        self.pretrain_dim = self.ent_embeddings.weight.shape[1]

        self.ent_embeddings.requires_grad_(False)
        self.rel_embeddings.requires_grad_(False)
        
        if adapter_hidden_dim is None:
            adapter_hidden_dim = max(self.pretrain_dim, self.emb_dim)
        
        self.adapter = self._build_mlp_adapter(
            input_dim=self.pretrain_dim,
            output_dim=self.emb_dim,
            hidden_dim=adapter_hidden_dim,
            num_layers=adapter_num_layers,
            dropout=adapter_dropout
        )
        
        print(f"Adapter MLP: {self.pretrain_dim} -> {adapter_hidden_dim} -> {self.emb_dim} ({adapter_num_layers} layers)")
    
    def _build_mlp_adapter(self, input_dim, output_dim, hidden_dim, num_layers=2, dropout=0.1):
        if num_layers == 1:
            return nn.Linear(input_dim, output_dim)
        
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        return nn.Sequential(*layers)

    def forward(self, triple_ids):
        
        if triple_ids is None:
            raise ValueError("triple_ids cannot be None. Please provide valid embedding_ids.")

        elif triple_ids.shape[1] == 32:  # 32 = 2 (head + relation) + 30 (candidates)
            batch_size = triple_ids.shape[0]
            batch_embeddings = []
            
            for batch_idx in range(batch_size):
                embedding_list = []
                for i, idx in enumerate(triple_ids[batch_idx]):
                    if i == 1:  # the second element is relation ID
                        embedding = self.rel_embeddings(idx)
                    else:  # the rest are entity IDs
                        embedding = self.ent_embeddings(idx)
                    embedding_list.append(embedding)
                
                batch_emb = torch.stack(embedding_list, dim=0)
                batch_embeddings.append(batch_emb)
            
            pretrain_embs = torch.stack(batch_embeddings, dim=0)
            prefix = self.adapter(pretrain_embs).reshape(-1, 32 * self.num_prefix, self.llm_dim)
            return prefix

        else:
            assert False, "triple_ids should have k+2 elements"
