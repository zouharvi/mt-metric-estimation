#!/usr/bin/env python3

import math
import sys
sys.path.append("src")
from figures import fig_utils
import matplotlib.pyplot as plt
import argparse
import json
import numpy as np

from matplotlib import gridspec


def get_color(val):
    if val > 0.2:
        return "black"
    else:
        return "white"


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d", "--data",
        default="computed/en_de_metric.jsonl",
    )
    args.add_argument(
        "-dh", "--data-human",
        default="computed/en_de_human_metric_brt.jsonl",
    )
    args = args.parse_args()

    PRETTY_NAME = {
        "bleu": "BLEU",
        "chrf": "ChrF",
        "ter": "TER",
        "meteor": "METEOR",
        "comet": "COMET",

        "tfidf": "LR TF-IDF",
        "lr_multi": "LR Multi",
        "conf_exp": "exp(conf.)",
        "conf_raw": "conf.",
        "len_raw": "|s|+|t|",
        "me_text": "ME text",
        "me_all": "ME all",
    }

    # load data with default dev sets sizes
    with open(args.data, "r") as f:
        data = [json.loads(x) for x in f.readlines()[:10000]]
    with open(args.data_human, "r") as f:
        data_human = [json.loads(x) for x in f.readlines()[:1000]]

    KEYS_X_HUMAN = [
        'conf', 'conf_exp',
        '|s|', '|t|', '|s|+|t|', '|s|-|t|', '|s|/|t|',
    ]
    KEYS_X_REST = [
        '|t_i|_var', 'conf_var', 'conf_exp_var',
        'h1_hx_bleu_avg', 'h1_hx_bleu_var', 'hx_hx_bleu_avg', 'hx_hx_bleu_var',
    ]

    KEYS_X = KEYS_X_HUMAN + KEYS_X_REST
    data_y = [sent["metrics"] for sent in data]
    data_human_y = [sent["metrics"] for sent in data_human]

    for sent in data + data_human:
        len_src = len(sent["src"].split())
        len_tgt = len(sent["tgts"][0][0].split())
        len_var = np.var([len(x[0].split()) for x in sent["tgts"]])
        sent["|t_i|_var"] = len_var
        sent["|s|"] = len_src
        sent["|t|"] = len_tgt
        sent["|s|+|t|"] = len_src + len_tgt
        # no abs is better
        # sent_x["||s|-|t||"] = abs(len_src - len_tgt)
        sent["|s|-|t|"] = len_src - len_tgt
        sent["|s|/|t|"] = len_src / len_tgt

    METRICS = ['bleu', 'chrf', 'ter', 'meteor', 'comet', "bleurt"]

    # compute metrics correlation with human (needed in the paper)
    data_zscore = [sent["zscore"] for sent in data_human_y]
    for metric in METRICS:
        data_metric = [sent[metric] for sent in data_human_y]
        print(f"zscore - {metric:<10}: {np.corrcoef(data_zscore, data_metric)[0,1]:.2%}")
    print("="*10)

    img1 = np.zeros((len(KEYS_X_HUMAN), len(METRICS) + 1))
    img2 = np.zeros((len(KEYS_X_REST), len(METRICS)))

    plt.figure(figsize=(10, 2.85))

    # the left table gets more width because it has an extra column
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.198, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # break into two subplots
    THRESHOLD_X = len(KEYS_X_HUMAN)

    for metric_i, metric in enumerate(METRICS):
        data_metric = [sent[metric] for sent in data_y]
        for key_x_i, key_x in enumerate(KEYS_X_HUMAN + KEYS_X_REST):
            # if key_x_i >= THRESHOLD_X:
            #     continue

            data_feature = [sent[key_x] for sent in data]
            corr = np.corrcoef(data_metric, data_feature)[0, 1]
            print(f"{metric:>10} - {key_x:<15}: {corr:>6.3f}")
            img = img1 if key_x_i < THRESHOLD_X else img2

            # color is based on absolute value but not the text!
            img[key_x_i % THRESHOLD_X, metric_i] = abs(corr)

            ax = ax1 if key_x_i < THRESHOLD_X else ax2
            ax.text(
                metric_i, key_x_i % THRESHOLD_X,
                f"{corr:.0%}",
                color=get_color(abs(corr)),
                ha="center", va="center"
            )
            if corr < 0:
                ax.text(
                    metric_i + 0.3, key_x_i % THRESHOLD_X + 0.3,
                    r"$\star$",
                    color=get_color(abs(corr)),
                )

    # put in human scores
    # zscore correlates less than score (against length based features)
    data_metric = [sent["zscore"] for sent in data_human_y]
    for key_x_i, key_x in enumerate(KEYS_X_HUMAN):
        data_feature = [sent[key_x] for sent in data_human]
        corr = np.corrcoef(data_metric, data_feature)[0, 1]
        print(f"{metric:>10} - {key_x:<15}: {corr:>6.3f}")
        img = img1 if key_x_i < THRESHOLD_X else img2

        # color is based on absolute value but not the text!
        img[key_x_i, len(METRICS)] = abs(corr)

        ax = ax1
        ax.text(
            len(METRICS), key_x_i % THRESHOLD_X,
            f"{corr:.2f}",
            color=get_color(abs(corr)),
            ha="center", va="center"
        )
        if corr < 0:
            ax.text(
                len(METRICS) + 0.3, key_x_i % THRESHOLD_X + 0.3,
                r"$\star$",
                color=get_color(abs(corr)),
            )

    cmap = plt.get_cmap("inferno")
    IMSHOW_KWARGS = {
        "cmap": cmap,
        "vmin": 0,
        "vmax": 0.45,
    }
    ax1.imshow(img1, aspect=0.6, **IMSHOW_KWARGS)
    ax2.imshow(img2, aspect=0.6, **IMSHOW_KWARGS)
    # plt.colorbar(cmap)

    for ax, keys_x, extra_metrics in zip(
        [ax1, ax2],
        [KEYS_X_HUMAN, KEYS_X_REST],
        [["zscore"], []],
    ):
        ax.set_xticks(
            list(range(len(METRICS + extra_metrics))),
            [
                fig_utils.PRETTY_NAME[m] if m in fig_utils.PRETTY_NAME else m
                for m in METRICS + extra_metrics
            ]
        )
        ax.xaxis.set_ticks_position("top")
        ax.set_yticks(
            list(range(len(keys_x))),
            [
                fig_utils.PRETTY_NAME[k] if k in fig_utils.PRETTY_NAME else k
                for k in keys_x
            ]
        )
    ax2.yaxis.set_ticks_position("right")

    plt.tight_layout(
        pad=0.1, w_pad=5,
        # make ticks visible
        rect=[0.04, 0, 0.96, 1],
    )
    plt.savefig("figures/feature_corr.pdf")
    plt.show()
