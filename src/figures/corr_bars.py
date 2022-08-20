#!/usr/bin/env python3

import matplotlib.pyplot as plt
import sys
sys.path.append("src")
from figures import fig_utils
import argparse
import json

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-bl", "--baseline-logfile",
        default="computed/de_en_somnorif_0.jsonl"
    )
    args.add_argument(
        "-ml", "--model-logfiles", nargs="+",
        default=[
            "computed/de_en_outroop_2.jsonl",
            "computed/de_en_outroop_3.jsonl",
        ],
    )
    args.add_argument(
        "-mn", "--model-names", nargs="+",
        default=[
            "src+hyp",
            "src+hyp+conf"
        ],
    )
    args = args.parse_args()
    data = []

    with open(args.baseline_logfile, "r") as f:
        data_b = [json.loads(line) for line in f.readlines()]
        data.append([x for x in data_b if x["model"] == "len_raw"][0])
        data.append([x for x in data_b if x["model"] == "conf_raw"][0])
        data.append([x for x in data_b if x["model"] == "conf_exp"][0])
        data.append([x for x in data_b if x["model"] == "lr_multi"][0])
        data_tfidf = [x for x in data_b if x["model"].startswith("tfidf_lr_")]
        model_best_tfidf = max(
            data_tfidf, key=lambda x: x["dev_corr"]
        )
        data.append(model_best_tfidf)

    for modelf, modeln in zip(args.model_logfiles, args.model_names):
        with open(modelf, "r") as f:
            data_m = [json.loads(line) for line in f.readlines()]
        model_best_epoch = max(
            data_m, key=lambda x: x["dev_corr"]
        )
        data.append({"model": modeln} | model_best_epoch)

    plt.bar(
        [x_i - 0.2 for x_i, x in enumerate(data)],
        [x["train_corr"] for x in data],
        tick_label=[
            ("\n" if x_i % 2 else "") + x["model"]
            for x_i, x in enumerate(data)
        ],
        width=0.4,
        label="Train correlation",
    )
    plt.bar(
        [x_i + 0.2 for x_i, x in enumerate(data)],
        [x["dev_corr"] for x in data],
        width=0.4,
        label="Dev correlation",
    )
    plt.legend()
    plt.show()

    print(data)
    exit()

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
