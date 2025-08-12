# !/usr/bin/env python
# -*- coding: UTF-8 -*-


import re
import json
import numpy as np
from collections import defaultdict
import argparse


def compute_MR(answer_list, prediction_lists, init_rank_list, init_list2name_list):
    ranks = []

    for answer, predictions, init_rank, init_list2name in zip(answer_list, prediction_lists, init_rank_list, init_list2name_list):
        if answer in predictions:
            rank = predictions.index(answer) + 1
        else:
            if answer not in init_list2name:
                rank = init_rank
            else:
                rank = max(len(predictions), init_rank)
        ranks.append(rank)

    mr = sum(ranks) / len(ranks)
    return mr


def compute_MRR(answer_list, prediction_lists, init_rank_list, init_list2name_list):
    reciprocal_ranks = []

    for answer, predictions, init_rank,  init_list2name in zip(answer_list, prediction_lists, init_rank_list, init_list2name_list):
        if answer in predictions:
            reciprocal_rank = 1 / (predictions.index(answer) + 1)
        else:
            if answer not in init_list2name:
                reciprocal_rank= 1 / init_rank
            else:
                reciprocal_rank = 1 / max(len(predictions), init_rank)

        reciprocal_ranks.append(reciprocal_rank)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return mrr


def compute_HR(answer_list, prediction_lists, k):
    hits = []
    for answer, predictions in zip(answer_list, prediction_lists):
        if answer in predictions[:k]:
            hit = 1
        else:
            hit = 0

        hits.append(hit)

    hr = sum(hits) / len(hits)

    return hr


def parse_prediction_string(pred_input):
    if isinstance(pred_input, list):
        return pred_input
    
    pred_str = pred_input
    pred_str = pred_str.strip().strip("'").strip('"')
    pred_str = pred_str.strip("[]")
    
    predictions = []
    if pred_str:
        pattern = r"'([^']*)'|\"([^\"]*)\"|([^,\s]+)"
        matches = re.findall(pattern, pred_str)
        
        for match in matches:
            entity = next((group for group in match if group), "")
            if entity.strip():
                predictions.append(entity.strip())
    
    return predictions


def evaluate(result_path, evaluation_path, dataset_name=None):

    answer_list = []
    prediction_lists = []
    
    with open(result_path, "r", encoding='utf-8') as f1:
        result_data = json.load(f1)
        
        for item in result_data:
            answer = item["answer"].strip()
            answer_list.append(answer)
            
            pred_input = item["predict"]
            predictions = parse_prediction_string(pred_input)
            prediction_lists.append(predictions)

    init_rank_list = []
    init_list2name_list = []
    test_data_path = f"./data/{dataset_name}/test.json"
    with open(test_data_path, "r", encoding='utf-8') as f2:
        test_data = json.load(f2)
        for item in test_data:
            init_rank_list.append(item["Init_rank"])
            init_list2name_list.append(item["Init_list2name"])

    min_length = min(len(answer_list), len(prediction_lists), len(init_rank_list))
    answer_list = answer_list[:min_length]
    prediction_lists = prediction_lists[:min_length]
    init_rank_list = init_rank_list[:min_length]
    init_list2name_list = init_list2name_list[:min_length]

    print(f"processed {min_length} samples\n")

    with open(evaluation_path, "w", encoding='utf-8') as fw:
        mr = compute_MR(answer_list, prediction_lists, init_rank_list, init_list2name_list)
        print(f"MR: {mr:.2f}")
        fw.write(f"MR: {mr:.2f}\n\n")

        mrr = compute_MRR(answer_list, prediction_lists, init_rank_list, init_list2name_list)
        print(f"MRR: {mrr * 100:.2f}")
        fw.write(f"MRR: {mrr * 100:.2f}\n\n")

        for k in [1, 3, 10]:
            hr = compute_HR(answer_list, prediction_lists, k)
            print(f"HR@{k}: {hr * 100:.2f}")
            fw.write(f"HR@{k}: {hr * 100:.2f}\n\n")
    
    return {
        "MR": mr,
        "MRR": mrr,
        "HR@1": compute_HR(answer_list, prediction_lists, 1),
        "HR@3": compute_HR(answer_list, prediction_lists, 3),
        "HR@10": compute_HR(answer_list, prediction_lists, 10)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate knowledge graph link prediction results')
    parser.add_argument('--result_path', type=str, default=None)
    parser.add_argument('--eval_path', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default=None)
    
    args = parser.parse_args()

    args.dataset_name = "UMLS"
    num_epoch = 1
    args.result_path = f"./log/res/{args.dataset_name}/results_epoch{num_epoch}.json"
    args.eval_path = f"./log/res/{args.dataset_name}/eval_epoch{num_epoch}.txt"
    
    evaluate(args.result_path, args.eval_path, args.dataset_name)
    
    print("\nevaluation completed!")
    print(f"results saved to: {args.eval_path}")
