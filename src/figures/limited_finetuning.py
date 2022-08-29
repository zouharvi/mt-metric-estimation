#!/usr/bin/env python3

import json
import argparse
import os
import sys
sys.path.append("src")
from figures import fig_utils
import matplotlib.pyplot as plt

# scp euler:/cluster/work/sachan/vilem/mt-metric-estimation/logs/en_de_windrose_1_*.jsonl logs/windrose/

def xtick_formatter(x):
    # hardcode overwrite
    if x == 14000:
        return "13k"

    if x % 1000 == 0:
        return f"{x//1000}k"
    else:
        return f"{x/1000}k"

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args = args.parse_args()

    data_h1 = []
    data_h2 = []
    data_h5 = []
    data_b = []

    plt.figure(figsize=(5, 3.3))

    # TODO: add "bleurt"
    METRICS = ["bleu", "chrf", "meteor", "comet", "ter", "zscore"]
    DSIZES = [500, 1000, 2000, 5000, 10000, 14000]

    data = {metric:[] for metric in METRICS}

    for metric in METRICS:
        for dsize in DSIZES:
            fpath = f"logs/windrose/en_de_windrose_1_{metric}_{dsize}.jsonl"
            if not os.path.isfile(fpath):
                continue
            with open(fpath, "r") as f:
                data_local = [json.loads(line) for line in f.readlines()]
                # TODO: abs?
                model_best_epoch = max(
                    data_local, key=lambda x: x["dev_corr"]
                )
                # TODO: abs?
                data[metric].append(model_best_epoch["dev_corr"])

    PLOT_KWARGS = {"marker": ".", "markersize": 13}

    for metric in METRICS:
        plt.plot(
            list(range(len(DSIZES)))[:len(data[metric])],
            [x for x in data[metric]],
            label=fig_utils.PRETTY_NAME[metric],
            **PLOT_KWARGS,
        )

    plt.xticks(
        list(range(len(DSIZES))),
        [xtick_formatter(x) for x in DSIZES]
    )
    plt.xlabel("Fine-tuning data size")
    plt.ylabel("z-score correlation")

    plt.legend(
        ncol=4, bbox_to_anchor=(0.45, 1.27), loc="upper center"
    )
    plt.tight_layout(rect=(0, 0, 0.99, 1.0), pad=0.1)
    plt.savefig("figures/limited_finetuning.pdf")
    plt.show()
