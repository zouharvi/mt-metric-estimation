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
    args.add_argument(
        "-mh2l", "--model-h2-logfiles", nargs="+",
        default=[
            "logs/en_de_dwile_2_1k.jsonl",
            "logs/en_de_dwile_2_5k.jsonl",
            "logs/en_de_dwile_2_10k.jsonl",
            "logs/en_de_dwile_2_50k.jsonl",
            "logs/en_de_dwile_2_100k.jsonl",
            "logs/en_de_dwile_2_500k.jsonl",
        ],
    )
    args.add_argument(
        "-mh5l", "--model-h5-logfiles", nargs="+",
        default=[
            "logs/en_de_dwile_3_1k.jsonl",
            "logs/en_de_dwile_3_5k.jsonl",
            "logs/en_de_dwile_3_10k.jsonl",
            "logs/en_de_dwile_3_50k.jsonl",
            "logs/en_de_dwile_3_100k.jsonl",
            # "logs/en_de_dwile_3_500k.jsonl",
        ],
    )
    args = args.parse_args()

    data_h1 = []
    data_h2 = []
    data_h5 = []
    data_b = []

    plt.figure(figsize=(5, 2.7))

    for f, n in zip(args.model_logfiles, args.nums):
        with open(f, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
            model_best_epoch = max(
                data, key=lambda x: x["dev_corr"]
            )
            data_h1.append(model_best_epoch["dev_corr"])


    for f, n in zip(args.model_h2_logfiles, args.nums):
        with open(f, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
            model_best_epoch = max(
                data, key=lambda x: x["dev_corr"]
            )
            data_h2.append(model_best_epoch["dev_corr"])

    for f, n in zip(args.model_h5_logfiles, args.nums):
        with open(f, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
            model_best_epoch = max(
                data, key=lambda x: x["dev_corr"]
            )
            data_h5.append(model_best_epoch["dev_corr"])

    for f, n in zip(args.baseline_logfiles, args.nums):
        with open(f, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
            model_lr_multi = [x for x in data if x["model"] == "lr_multi"][0]
            data_b.append(model_lr_multi["dev_corr"])

    PLOT_KWARGS = {"marker": ".", "markersize": 13}

    plt.plot(
        list(range(len(args.nums)))[:len(data_h5)],
        [x for x in data_h5],
        label="ME all (H5)",
        **PLOT_KWARGS,
    )
    plt.plot(
        list(range(len(args.nums)))[:len(data_h2)],
        [x for x in data_h2],
        label="ME all (H2)",
        **PLOT_KWARGS,
    )
    plt.plot(
        list(range(len(args.nums))),
        [x for x in data_h1],
        label="ME all (H1)",
        **PLOT_KWARGS,
    )
    plt.plot(
        list(range(len(args.nums))),
        [x for x in data_b],
        label="LR Multi (auth.)",
        linestyle=":",
        color="black",
        **PLOT_KWARGS,
    )

    plt.xticks(
        list(range(len(args.nums))),
        [f"{x//1000}k" for x in args.nums]
    )
    plt.xlabel("Authentic data size")
    plt.ylabel("BLEU correlation")

    plt.legend()
    plt.tight_layout(pad=0.2)
    plt.savefig("figures/limited_data.pdf")
    plt.show()
