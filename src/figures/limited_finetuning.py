#!/usr/bin/env python3

import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import sys
sys.path.append("src")
from figures import fig_utils

# scp euler:/cluster/work/sachan/vilem/mt-metric-estimation/logs/en_de_windrose_1_*.jsonl logs/windrose/


def conf_interval(xs):
    if len(xs) <= 2:
        return 0
    interval = st.t.interval(
        confidence=0.95, df=len(xs) - 1,
        loc=np.mean(xs),
        scale=st.sem(xs)
    )
    # return radius
    radius = (interval[1] - interval[0]) / 2
    # if radius > 0.2:
    #     print(xs)
    return radius


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
    METRICS = ["bleu", "bleurt", "chrf", "meteor", "comet", "ter", "zscore"]
    DSIZES = [500, 1000, 2000, 5000, 10000, 14000]
    # add both versions because of an early bug
    SEED_SUFFIXES = (
        [""]
        + ["_s{" + str(i) + "}" for i in [1, 2, 3, 4, 5]]
        + ["_s" + str(i) + "" for i in [1, 2, 3, 4, 5]]
        + ["_" + str(i) + "" for i in [1, 2, 3, 4, 5]]
    )

    data = {metric: {s: [] for s in DSIZES} for metric in METRICS}

    for metric in METRICS:
        for dsize in DSIZES:
            for suffix in SEED_SUFFIXES:
                fpath = f"logs/windrose/en_de_windrose_1_{metric}_{dsize}{suffix}.jsonl"
                if not os.path.isfile(fpath):
                    continue
                with open(fpath, "r") as f:
                    data_local = [json.loads(line) for line in f.readlines()]
                    # TODO: abs?
                    model_best_epoch = max(
                        list(enumerate(data_local)), key=lambda x: abs(x[1]["dev_corr"])
                    )
                    # print(model_best_epoch[0])
                    # TODO: abs?
                    data[metric][dsize].append(abs(model_best_epoch[1]["dev_corr"]))

    PLOT_KWARGS = {"marker": ".", "markersize": 13}

    for metric_i, metric in enumerate(METRICS):
        xticks = list(range(len(DSIZES)))[:len(data[metric])]
        yvals = [np.average(data[metric][dsize]) for dsize in DSIZES]
        plt.plot(
            xticks, yvals,
            label=fig_utils.PRETTY_NAME[metric],
            color=fig_utils.COLORS[metric_i],
            **PLOT_KWARGS,
        )
        if metric in {"zscore", "ter"}:
            plt.errorbar(
                x=xticks, y=yvals,
                yerr=[conf_interval(data[metric][dsize]) for dsize in DSIZES],
                color=fig_utils.COLORS[metric_i],
                elinewidth=1,
                capsize=2,
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
