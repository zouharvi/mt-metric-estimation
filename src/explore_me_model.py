#!/usr/bin/env python3


import pickle
import json
import argparse
import sys
sys.path.append("src")
import me_zoo
import utils

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d", "--data",
        default="computed/en_de_human_metric.jsonl"
    )
    args.add_argument("-mp", "--model-path", default="1")
    args.add_argument("-dn", "--data-n", type=int, default=None)
    args.add_argument("-bp", "--bpe-path",
                      default="models/models/bpe_news_500k_h1.pkl")
    args.add_argument("-m", "--model", default="1hd75b10lin")
    args.add_argument("-f", "--fusion", type=int, default=1)
    args.add_argument("--metric", default="bleu")
    args = args.parse_args()

    model, vocab_size = me_zoo.get_model(args)

    with open(args.data, "r") as f:
        data = [json.loads(x) for x in f.readlines()][:args.data_n]

    data = [
        sent | {
            "src+hyp": sent["src"] + " [SEP] " + sent["tgts"][0][0],
            "hyp": sent["tgts"][0][0],
        }
        for sent in data
    ]

    with open(args.bpe_path, "rb") as f:
        encoder = pickle.load(f)

    data_bpe = encoder.transform([x["src+hyp"] for x in data])
    data = [
        {"src+hyp_bpe": sent_bpe} | sent
        for sent, sent_bpe in zip(data, data_bpe)
    ]

    _, y_preds = model.eval_dev(data, args.metric)

    pred_all = [
        (y_pred, sent["metrics"][args.metric], sent)
        for y_pred, sent in zip(y_preds, data)
        # take only short sentences
        if len(sent["src"].split()) < 10
    ]

    # sort by difference in prediction
    pred_all.sort(key=lambda x: abs(x[0] - x[1]))

    def print_example(x):
        sent = x[2]
        print(
            f"{x[0]:>5.2f} | {x[1]:>5.2f} | {sent['src']} | {sent['ref']} | {sent['tgts'][0][0]}")

    print_example(pred_all[0])
    print_example(pred_all[5])
    print_example(pred_all[10])
    print_example(pred_all[15])
    print_example(pred_all[20])
    print_example(pred_all[25])
    print()
    print_example(pred_all[-0])
    print_example(pred_all[-5])
    print_example(pred_all[-10])
    print_example(pred_all[-15])
    print_example(pred_all[-20])
    print_example(pred_all[-25])