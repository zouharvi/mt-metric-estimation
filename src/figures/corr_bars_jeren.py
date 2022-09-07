#!/usr/bin/env python3

from collections import defaultdict
import matplotlib.pyplot as plt
import sys
sys.path.append("src")
from figures import fig_utils
import argparse
import json

def pretty_lang_dir(direction):
    return direction[:2].capitalize() + "${\\rightarrow}$" + direction[3:].capitalize()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    # args.add_argument()
    args = args.parse_args()

    METRICS = ["bleu", "bleurt", "chrf", "meteor", "comet", "ter"]
    DIRECTIONS = [
        "en_de", "en_cs", "en_fr", "ru_en", "en_hi", "en_zh", "de_pl",
        "de_en", "cs_en", "fr_en", "en_ru", "hi_en", "zh_en", "pl_de",
    ]

    DIR_THRESHOLD = 7

    data1 = defaultdict(list)
    data2 = defaultdict(list)
    plt.figure(figsize=(5, 6))

    # possible future bug: this only works because comet-qe is at the beginning
    for metric_i, metric in enumerate(METRICS):
        for direction_i, direction in enumerate(DIRECTIONS):
            with open(f"logs/jeren/jeren_{direction}_{metric}.jsonl", "r") as f:
                data_local = [json.loads(line) for line in f.readlines()]
                model_best_epoch = max(
                    data_local, key=lambda x: x["dev_corr"]
                )
                if direction_i < DIR_THRESHOLD:
                    data1[metric].append(model_best_epoch)
                else:
                    data2[metric].append(model_best_epoch)

    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    for ax, data in zip([ax1, ax2], [data1, data2]):
        for metric_i, metric in enumerate(METRICS):
            data_local = data[metric]
            ax.bar(
                [
                    x_i + metric_i / (len(METRICS) + 1.5)
                    for x_i, x in enumerate(data_local)
                ],
                [100 * abs(x["dev_corr"]) for x in data_local],
                width=1 / (len(METRICS) + 1.5),
                label=fig_utils.PRETTY_NAME[metric],
                edgecolor="black",
                linewidth=1.5,
                hatch="",
            )

    DIRECTIONS = [pretty_lang_dir(x) for x in DIRECTIONS]
    DIRECTIONS1 = DIRECTIONS[:DIR_THRESHOLD]
    DIRECTIONS2 = DIRECTIONS[DIR_THRESHOLD:]

    ax1.set_ylim(None, 90)
    ax2.set_ylim(None, 90)

    ax1.set_ylabel("Correlation (%)")
    ax2.set_ylabel("Correlation (%)")
    ax1.set_xticks([x_i + 0.32 for x_i, x in enumerate(DIRECTIONS1)])
    ax1.set_xticklabels(DIRECTIONS1)
    ax2.set_xticks([x_i + 0.32 for x_i, x in enumerate(DIRECTIONS2)])
    ax2.set_xticklabels(DIRECTIONS2)

    # reset default to ax1
    ax1 = plt.subplot(2, 1, 1)
    plt.legend(
        ncol=3, bbox_to_anchor=(0.45, 1.25), loc="upper center",
    )
    plt.tight_layout(rect=(-0.00, 0, 0.99, 0.98), pad=0.1)
    plt.savefig("figures/baseline_comparison_otherlangs.pdf")
    plt.show()
