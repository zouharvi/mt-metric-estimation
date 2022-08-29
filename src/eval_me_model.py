#!/usr/bin/env python3

import json
import argparse
import pickle
import numpy as np
import sys
sys.path.append("src")
import me_zoo

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d", "--data",
        default="computed/en_de_human_metric_brt.jsonl"
    )
    args.add_argument(
        "-od", "--output-data",
        default=None
    )
    args.add_argument(
        "-mp", "--model-load-path",
        default="models/en_de_outroop_25_bleu_bleu_s.pt"
    )
    args.add_argument("-m", "--model", default="1hd75b10lin")
    args.add_argument("-f", "--fusion", type=int, default=2)
    args.add_argument("-dn", "--data-n", type=int, default=None)
    args.add_argument(
        "-lb", "--load-bpe",
        default="models/bpe_news_500k_h1.pkl"
    )
    args.add_argument("--metric", default="bleu")
    args = args.parse_args()

    model, vocab_size = me_zoo.get_model(args)

    with open(args.data, "r") as f:
        data = [json.loads(x) for x in f.readlines()][:args.data_n]

    if "human" in args.data and args.data_n != 1000:
        print(
            "You're using the human data but your dev-n is not 1k as described in the paper"
        )
        exit()
    if "human" not in args.data and args.data_n != 10000:
        print(
            "You're not using the human data but your dev-n is not 10k as described in the paper"
        )
        exit()

    data = [
        sent | {
            "src+hyp": sent["src"] + " [SEP] " + sent["tgts"][0][0],
            "hyp": sent["tgts"][0][0],
        }
        for sent in data
    ]

    with open(args.load_bpe, "rb") as f:
        encoder = pickle.load(f)

    data_bpe = encoder.transform([x["src+hyp"] for x in data])
    data = [
        {"src+hyp_bpe": sent_bpe} | sent
        for sent, sent_bpe in zip(data, data_bpe)
    ]

    print(f"Evaluating model {args.model} with fusion {args.fusion}")
    _, y_pred = model.eval_dev(
        data,
        metric=args.metric,
    )
    y_true = [sent["metrics"][args.metric] for sent in data]

    corr = np.corrcoef(y_pred, y_true)[0, 1]

    print(f"Correlation with {args.metric} is {corr:.2%}")

    if args.output_data is not None:
        fout = open(args.output_data, "w")
        for sent, y_pred_val in zip(data, y_pred):
            if "metrics_pred" not in sent:
                sent["metrics_pred"] = {}
            sent.pop("src+hyp")
            sent.pop("src+hyp_bpe")
            sent["metrics_pred"][args.metric] = y_pred_val
            fout.write(json.dumps(sent, ensure_ascii=False) + "\n")