#!/usr/bin/env python3

from collections import defaultdict
import matplotlib.pyplot as plt
import sys
sys.path.append("src")
from figures import fig_utils
import argparse
import json

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-bl", "--baseline-logfiles", nargs="+",
        default=[
            "logs/de_en_somnorif_3_bleu.jsonl",
            "logs/de_en_somnorif_3_chrf.jsonl",
            "logs/de_en_somnorif_3_ter.jsonl",
            "logs/de_en_somnorif_3_meteor.jsonl",
            "logs/de_en_somnorif_3_comet.jsonl",
            # TODO add human
        ]
    )
    args.add_argument(
        "-mla", "--model-logfiles-all", nargs="+",
        default=[
            "logs/de_en_outroop_19_bleu.jsonl",  # bleu
            "logs/de_en_outroop_19_chrf.jsonl",  # bleu
            "logs/de_en_outroop_19_bleu.jsonl",  # bleu
            "logs/de_en_outroop_19_meteor.jsonl",  # bleu
            "logs/de_en_outroop_19_bleu.jsonl",  # bleu
        ],
    )
    args.add_argument(
        "-mlt", "--model-logfiles-text", nargs="+",
        default=[
            "logs/de_en_outroop_19_bleu.jsonl",  # bleu, incorrect
        ],
    )
    args = args.parse_args()

    METRICS = ["bleu", "chrf", "ter", "meteor", "comet"]
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
        "me_all": "ME all",
    }
    data = defaultdict(list)

    plt.figure(figsize=(5, 3))

    for f, metric in zip(args.baseline_logfiles, METRICS):
        with open(f, "r") as f:
            data_b = [json.loads(line) for line in f.readlines()]
            data[metric].append(
                [x for x in data_b if x["model"] == "len_raw"][0]
            )
            data[metric].append(
                [x for x in data_b if x["model"] == "conf_raw"][0]
            )
            data[metric].append(
                [x for x in data_b if x["model"] == "conf_exp"][0]
            )
            data_tfidf = [
                x for x in data_b if x["model"].startswith("tfidf_lr_")]
            model_best_tfidf = max(
                data_tfidf, key=lambda x: x["dev_corr"]
            )
            model_best_tfidf["model"] = "tfidf"
            data[metric].append(model_best_tfidf)
            data[metric].append(
                [x for x in data_b if x["model"] == "lr_multi"][0]
            )


    for f, metric in zip(args.model_logfiles_all, METRICS):
        with open(f, "r") as f:
            data_m = [json.loads(line) for line in f.readlines()]
            model_best_epoch = max(
                data_m, key=lambda x: x["dev_corr"]
            )
            data[metric].append({"model": "me_all"} | model_best_epoch)

    for metric_i, metric in enumerate(METRICS):
        data_local = data[metric]
        plt.bar(
            [x_i + metric_i / (len(METRICS) + 1)
             for x_i, x in enumerate(data_local)],
            [abs(x["dev_corr"]) for x in data_local],
            tick_label=[
                ("\n" if x_i % 2 else "") + PRETTY_NAME[x["model"]]
                for x_i, x in enumerate(data_local)
            ] if metric_i == 2 else None,
            width=1 / (len(METRICS) + 1),
            label=PRETTY_NAME[metric],
            edgecolor="black",
            linewidth=1.5,
        )

    plt.legend(
        ncol=3, bbox_to_anchor=(0.5, 1.3), loc="upper center"
    )
    plt.tight_layout(rect=(0, 0, 1, 1.02), pad=0.1)
    plt.show()
