#!/usr/bin/env python3

import math
import sys
sys.path.append("src")
from figures import fig_utils
import matplotlib.pyplot as plt
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
        "-d", "--data",
        default="computed/en_de_metric_pred.jsonl",
    )
    args.add_argument(
        "-dh", "--data-human",
        default="computed/en_de_human_metric_pred.jsonl",
    )
    args = args.parse_args()

    # load data with default dev sets sizes
    with open(args.data, "r") as f:
        data = [json.loads(x) for x in f.readlines()[:10000]]
    with open(args.data_human, "r") as f:
        data_human = [json.loads(x) for x in f.readlines()[:1000]]

    METRICS = ['bleu', 'bleurt', 'chrf', 'ter', 'meteor', 'comet']
    PLT_KWARGS = {
        "showmeans": False,
        "showmedians": False,
        "showextrema": False,
        "widths": 0.95,
    }

    data_y = [
        [get_metric_val(sent["metrics"], metric) for sent in data]
        for metric in METRICS
    ]
    data_y_pred = [
        [get_metric_val(sent["metrics_pred"], metric) for sent in data]
        for metric in METRICS
    ]
    METRICS += ["zscore"]
    data_human_y = [
        [get_metric_val(sent["metrics"], metric) for sent in data_human]
        for metric in METRICS
    ]
    data_human_y_pred = [
        [get_metric_val(sent["metrics_pred"], metric) for sent in data_human]
        for metric in METRICS
    ]

    plt.figure(figsize=(5, 3))
    v1 = plt.violinplot(
        data_y,
        **PLT_KWARGS
    )
    v1pred = plt.violinplot(
        data_y_pred,
        **PLT_KWARGS
    )
    v2 = plt.violinplot(
        data_human_y,
        **PLT_KWARGS
    )
    v2pred = plt.violinplot(
        data_human_y_pred,
        **PLT_KWARGS
    )

    label_handles = {}

    for (violin, dir, ispred) in zip(
        [v1, v2, v1pred, v2pred],
        ["left", "right", "left", "right"],
        [False, False, True, True],
    ):
        for pc in violin["bodies"]:
            # get the center
            m = np.mean(pc.get_paths()[0].vertices[:, 0])
            # modify the paths to not go further right than the center
            if dir == "left":
                pc.get_paths()[0].vertices[:, 0] = np.clip(
                    pc.get_paths()[0].vertices[:, 0],
                    -np.inf, m
                )
            elif dir == "right":
                pc.get_paths()[0].vertices[:, 0] = np.clip(
                    pc.get_paths()[0].vertices[:, 0],
                    m, np.inf
                )

            # styling
            if dir == "left":
                pc.set_facecolor(fig_utils.COLORS[0])
            elif dir == "right":
                pc.set_facecolor(fig_utils.COLORS[1])

            pc.set_edgecolor('black')
            pc.set_linewidth(1.2)
            pc.set_aa(True)
            pc.set_alpha(0.9)

            if not ispred:
                pc.set_hatch("....")
                if dir == "right":
                    label_handles["QE true"] = pc
                else:
                    label_handles["ME true"] = pc
            else:
                if dir == "right":
                    label_handles["QE pred."] = pc
                else:
                    label_handles["ME pred."] = pc

    plt.legend(
        handles=label_handles.values(),
        labels=label_handles.keys(),
        ncol=2,
        loc="lower center",
        borderpad=0.35,
        columnspacing=0.6,
    )
    plt.ylim(-1, 1.5)
    plt.xticks(
        range(1, len(METRICS) + 1),
        [fig_utils.PRETTY_NAME[m] for m in METRICS],
    )
    plt.tight_layout(
        pad=0.15
    )
    plt.savefig("figures/metric_violin.pdf")
    plt.show()
