#!/usr/bin/env python3

import matplotlib.pyplot as plt
import sys
sys.path.append("src")
from figures import fig_utils
import argparse
import json

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-l", "--logfile",
                      default="computed/de_en_outroop.jsonl")
    args = args.parse_args()

    with open(args.logfile, "r") as f:
        data = [json.loads(line) for line in f.readlines()]

    xticks = list(range(1, len(data) + 1))

    ax1 = plt.gca()
    ax2 = ax1.twinx()

    lns1t = ax1.plot(
        xticks,
        [x["train_loss"] for x in data],
        label="Train MSE",
        linestyle=":", color=fig_utils.COLORS[0],
        marker="x"
    )
    lns1d = ax1.plot(
        xticks,
        [x["dev_loss"] for x in data],
        label="Dev MSE",
        linestyle=":", color=fig_utils.COLORS[1],
        marker="x"
    )
    lns2t = ax2.plot(
        xticks,
        [x["train_corr"] for x in data],
        label="Train correlation",
        linestyle="-", color=fig_utils.COLORS[0],
        marker="o"
    )
    lns2d = ax2.plot(
        xticks,
        [x["dev_corr"] for x in data],
        label="Dev correlation",
        linestyle="-", color=fig_utils.COLORS[1],
        marker="o"
    )

    # ax1.set_xticks(xticks)
    # ax1.set_xticklabels(xticks)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("MSE against BLEU")
    ax2.set_ylabel("Correlation with BLEU")

    lns_all = lns1t + lns1d + lns2t + lns2d
    lbs_all = [l.get_label() for l in lns_all]
    ax1.legend(
        lns_all, lbs_all, loc="upper center",
        bbox_to_anchor=(0.5, 1.2), ncol=2
    )

    plt.tight_layout(rect=[0, 0, 1, 1.02])
    plt.show()
