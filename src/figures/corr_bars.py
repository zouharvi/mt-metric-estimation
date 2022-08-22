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
        default="logs/de_en_somnorif_2.jsonl"
    )
    args.add_argument(
        "-ml", "--model-logfiles", nargs="+",
        default=[
            # "logs/de_en_outroop_2.jsonl",
            "logs/de_en_outroop_15.jsonl",
        ],
    )
    args.add_argument(
        "-mn", "--model-names", nargs="+",
        default=[
            # "src+hyp",
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
        # data.append([x for x in data_b if x["model"] == "lr_multi"][0])
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
