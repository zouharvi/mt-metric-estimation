#!/usr/bin/env python3

import json
import argparse
import sys
sys.path.append("src")
from figures import fig_utils
from collections import defaultdict
import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-bl", "--baseline-logfiles", nargs="+",
        default=[
            "logs/en_de_dwile_1_1k.jsonl",
            "logs/en_de_dwile_1_5k.jsonl",
            "logs/en_de_dwile_1_10k.jsonl",
            "logs/en_de_dwile_1_50k.jsonl",
            "logs/en_de_dwile_1_100k.jsonl",
            "logs/en_de_dwile_1_500k.jsonl",
        ],
    )
    args.add_argument(
        "-n", "--nums", nargs="+",
        default=[
            1000, 5000, 10000, 50000, 100000, 500000
        ]
    )
    args.add_argument(
        "-ml", "--model-logfiles", nargs="+",
        default=[
            "logs/en_de_dwile_0_1k.jsonl",
            "logs/en_de_dwile_0_5k.jsonl",
            "logs/en_de_dwile_0_10k.jsonl",
            "logs/en_de_dwile_0_50k.jsonl",
            "logs/en_de_dwile_0_100k.jsonl",
            "logs/en_de_dwile_0_500k.jsonl",
        ],
    )
    args = args.parse_args()

    data_me = []
    data_b = []

    plt.figure(figsize=(5, 2))

    for f, n in zip(args.model_logfiles, args.nums):
        with open(f, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
            model_best_epoch = max(
                data, key=lambda x: x["dev_corr"]
            )
            data_me.append(model_best_epoch["dev_corr"])

    for f, n in zip(args.baseline_logfiles, args.nums):
        with open(f, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
            model_lr_multi = [x for x in data if x["model"] == "lr_multi"][0]
            data_b.append(model_lr_multi["dev_corr"])

    PLOT_KWARGS = {"marker": "."}

    plt.plot(
        list(range(len(args.nums))),
        [x for x in data_me],
        label="ME all (auth.)",
        **PLOT_KWARGS,
    )
    plt.plot(
        list(range(len(args.nums))),
        [x for x in data_b],
        label="LR Multi (auth.)",
        **PLOT_KWARGS,
    )

    plt.xticks(
        list(range(len(args.nums))),
        [f"{x//1000}k" for x in args.nums]
    )

    plt.legend()
    plt.tight_layout(pad=0.1)
    plt.savefig("figures/limited_data.pdf")
    plt.show()
