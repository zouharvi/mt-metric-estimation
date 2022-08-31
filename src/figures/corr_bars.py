#!/usr/bin/env python3

from collections import defaultdict
import matplotlib.pyplot as plt
import sys
sys.path.append("src")
from figures import fig_utils
import argparse
import json

# scp euler:/cluster/work/sachan/vilem/mt-metric-estimation/logs/en_de_outroop_25_ter*.jsonl logs/
# scp euler:/cluster/work/sachan/vilem/mt-metric-estimation/logs/en_de_zepole*.jsonl logs/

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--baseline-logfiles", nargs="+",
        # NOTE: run this locally 
        default=[
            # 10k
            "logs/en_de_somnorif_4_bleu.jsonl",
            "logs/en_de_somnorif_4_bleurt.jsonl",
            "logs/en_de_somnorif_4_chrf.jsonl",
            "logs/en_de_somnorif_4_meteor.jsonl",
            "logs/en_de_somnorif_4_comet.jsonl",
            "logs/en_de_somnorif_4_ter.jsonl",
            # 1k
            "logs/en_de_somnorif_4_zscore.jsonl",
        ]
    )
    args.add_argument(
        "--mbert-logfiles", nargs="+",
        default=[
            # 10k
            "logs/en_de_zepole_1_bleu.jsonl",
            "logs/en_de_zepole_1_bleurt.jsonl",
            "logs/en_de_zepole_1_chrf.jsonl",
            "logs/en_de_zepole_1_meteor.jsonl",
            "logs/en_de_zepole_1_comet.jsonl",
            "logs/en_de_zepole_1_ter.jsonl",
            # 1k
            "logs/en_de_zepole_1_zscore.jsonl",
        ],
    )
    args.add_argument(
        "--model-logfiles-all", nargs="+",
        default=[
            # 10k
            "logs/en_de_outroop_25_bleu_bleu.jsonl",
            "logs/en_de_outroop_25_bleurt_bleurt.jsonl",
            "logs/en_de_outroop_25_chrf_chrf.jsonl",
            "logs/en_de_outroop_25_meteor_meteor.jsonl",
            "logs/en_de_outroop_25_comet_comet.jsonl",
            "logs/en_de_outroop_25_ter_ter_unscaled.jsonl",
            # 1k
            "logs/en_de_outroop_23_zscore_zscore_r_news.jsonl",
        ],
    )
    args.add_argument(
        "--model-logfiles-text", nargs="+",
        default=[
            # 10k
            "logs/en_de_outroop_24_bleu_bleu.jsonl",
            "logs/en_de_outroop_24_bleurt_bleurt.jsonl",
            "logs/en_de_outroop_24_chrf_chrf.jsonl",
            "logs/en_de_outroop_24_meteor_meteor.jsonl",
            "logs/en_de_outroop_24_comet_comet.jsonl",
            "logs/en_de_outroop_24_ter_ter.jsonl",
            # 1k
            "logs/en_de_outroop_24_zscore_zscore_r_news.jsonl",
        ],
    )
    args.add_argument(
        "-cl", "--comet-logfile", default="logs/en_de_hopsack_0_edited.jsonl",
    )
    args = args.parse_args()

    METRICS = ["bleu", "bleurt", "chrf", "meteor", "comet", "ter", "zscore"]

    data = defaultdict(list)

    plt.figure(figsize=(5, 3))

    # possible future bug: this only works because comet-qe is at the beginning
    with open(args.comet_logfile, "r") as f:
        data_comet = [json.loads(line) for line in f.readlines()]
    for metric in METRICS:
        data[metric].append(
            [x for x in data_comet if x["metric"] == metric][0]
            )


    for f, metric in zip(args.baseline_logfiles, METRICS):
        with open(f, "r") as f:
            data_b = [json.loads(line) for line in f.readlines()]
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

    for f, metric in zip(args.mbert_logfiles, METRICS):
        with open(f, "r") as f:
            data_m = [json.loads(line) for line in f.readlines()]
            model_best_epoch = max(
                data_m, key=lambda x: x["dev_corr"]
            )
            data[metric].append({"model": "mbert"} | model_best_epoch)

    for fn, metric in zip(args.model_logfiles_all, METRICS):
        with open(fn, "r") as f:
            data_m = [json.loads(line) for line in f.readlines()]
            model_best_epoch = max(
                data_m, key=lambda x: x["dev_corr"]
            )
            print(f"{fn}: {model_best_epoch['dev_corr']:.2%}")
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
            [
                x_i + metric_i / (len(METRICS) + 1.5)
                for x_i, x in enumerate(data_local)
            ],
            [100 * abs(x["dev_corr"]) for x in data_local],
            tick_label=[
                # ("\n" if x_i % 2 else "") +
                fig_utils.PRETTY_NAME[x["model"]]
                for x_i, x in enumerate(data_local)
            ] if metric_i == 2 else None,
            width=1 / (len(METRICS) + 1.5),
            label=fig_utils.PRETTY_NAME[metric],
            edgecolor="black",
            linewidth=1.5,
            hatch="////" if metric == "zscore" else "",
        )

    plt.vlines(
        x=[0.835, 2.845], ymin=0, ymax=65,
        linestyle=":", color="black",
        linewidth=1
    )
    plt.ylim(None, 63)
    plt.ylabel("Correlation (%)")

    plt.legend(
        ncol=4, bbox_to_anchor=(0.45, 1.27), loc="upper center"
    )
    plt.tight_layout(rect=(-0.015, 0, 1.03, 1.0), pad=0.1)
    plt.savefig("figures/baseline_comparison.pdf")
    plt.show()
