#!/usr/bin/env python3

import sys
sys.path.append("src")
import argparse
import json
import numpy as np


def get_color(val):
    if val > 0.2:
        return "black"
    else:
        return "white"


def get_metric_val(sent, metric):
    if metric == "ter":
        val = sent["ter"] / 100
    else:
        val = sent[metric]

    return np.clip(val, -1, 1.5)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-dh", "--data-human",
        default="computed/en_de_human_metric_ft.jsonl",
    )
    args = args.parse_args()

    # load data with default dev sets sizes
    with open(args.data_human, "r") as f:
        data = [json.loads(x) for x in f.readlines()]

    METRICS = ['bleu', 'bleurt', 'chrf', 'ter', 'meteor', 'comet']
    SCORE_TYPES = ["zscore", "score"]
    for score_type in SCORE_TYPES:
        score_type_corrs = []
        for metric in METRICS:
            data_x = [x["metrics"][metric] for x in data]
            data_y = [x["metrics"][score_type] for x in data]
            corr = np.corrcoef(data_x, data_y)[0,1]
            print(f"{metric:>10}-{score_type:<10} {corr:.2%}")
            score_type_corrs.append(np.abs(corr))
        print(f"AVG {np.average(score_type_corrs):.2%}")
        print()