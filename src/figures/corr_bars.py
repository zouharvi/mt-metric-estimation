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
        # TODO: change to en-de
        default=[
            "logs/en_de_somnorif_4_chrf.jsonl",
            "logs/en_de_somnorif_4_bleu.jsonl",
            "logs/en_de_somnorif_4_comet.jsonl",
            "logs/en_de_somnorif_4_meteor.jsonl",
            "logs/en_de_somnorif_4_ter.jsonl",
            "logs/en_de_somnorif_4_zscore.jsonl",
        ]
    )
    args.add_argument(
        "-mlad", "--model-logfiles-all", nargs="+",
        # TODO: currently just human features
        # results are en-de, the filenames are wrong
        default=[
            # 10k
            "logs/de_en_outroop_20_chrf.jsonl",
            "logs/de_en_outroop_20_bleu.jsonl",
            "logs/de_en_outroop_20_comet.jsonl",
            "logs/de_en_outroop_20_meteor.jsonl",
            "logs/de_en_outroop_20_ter.jsonl",
            # 1k
            "logs/de_en_outroop_23_zscore_zscore.jsonl",
        ],
    )
    args.add_argument(
        "-mlt", "--model-logfiles-text", nargs="+",
        # TODO: change to text-only
        # results are en-de, the filenames are wrong
        default=[
            # 10k
            "logs/de_en_outroop_22_chrf.jsonl",
            "logs/de_en_outroop_22_bleu.jsonl",
            "logs/de_en_outroop_22_comet.jsonl",
            "logs/de_en_outroop_22_meteor.jsonl",
            "logs/de_en_outroop_22_ter.jsonl",
            # 1k
            "logs/en_de_outroop_24_zscore_zscore.jsonl",
        ],
    )
    args.add_argument(
        "-cl", "--comet-logfile", default="logs/en_de_hopsack_0_edited.jsonl",
    )
    args = args.parse_args()

    METRICS = ["bleu", "chrf", "ter", "meteor", "comet", "zscore"]

    data = defaultdict(list)

    plt.figure(figsize=(5, 2.7))

    # possible future bug: this only works because comet-qe is at the beginning
    with open(args.comet_logfile, "r") as f:
        data_comet = [json.loads(line) for line in f.readlines()]
    for metric in METRICS:
        data[metric].append([x for x in data_comet if x["metric"] == metric][0])

    for f, metric in zip(args.baseline_logfiles, METRICS):
        with open(f, "r") as f:
            data_b = [json.loads(line) for line in f.readlines()]
            # individual features are going to be in a different figures
            # data[metric].append(
            #     [x for x in data_b if x["model"] == "len_raw"][0]
            # )
            # data[metric].append(
            #     [x for x in data_b if x["model"] == "conf_raw"][0]
            # )
            # data[metric].append(
            #     [x for x in data_b if x["model"] == "conf_exp"][0]
            # )
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

    for f, metric in zip(args.model_logfiles_text, METRICS):
        with open(f, "r") as f:
            data_m = [json.loads(line) for line in f.readlines()]
            model_best_epoch = max(
                data_m, key=lambda x: x["dev_corr"]
            )
            data[metric].append({"model": "me_text"} | model_best_epoch)


    for metric_i, metric in enumerate(METRICS):
        data_local = data[metric]
        plt.bar(
            [x_i + metric_i / (len(METRICS) + 1.5)
             for x_i, x in enumerate(data_local)],
            [abs(x["dev_corr"]) for x in data_local],
            tick_label=[
                # ("\n" if x_i % 2 else "") + 
                fig_utils.PRETTY_NAME[x["model"]]
                for x_i, x in enumerate(data_local)
            ] if metric_i == 2 else None,
            width=1 / (len(METRICS) + 1.5),
            label=fig_utils.PRETTY_NAME[metric],
            edgecolor="black",
            linewidth=1.5,
        )

    plt.vlines(
        x= [0.835, 2.835], ymin=0, ymax=0.65,
        linestyle=":", color="black",
    )
    plt.ylim(None, 0.65)

    plt.legend(
        ncol=3, bbox_to_anchor=(0.5, 1.3), loc="upper center"
    )
    plt.tight_layout(rect=(0, 0, 1, 1.02), pad=0.1)
    plt.savefig("figures/baseline_comparison.pdf")
    plt.show()
