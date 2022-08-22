#!/usr/bin/env python3

import sys
sys.path.append("src")
import utils
import argparse
import json
import csv
import os
import me_zoo

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--data", default="computed/de_en_metric.csv")
    args.add_argument("-m", "--model", default="1")
    args.add_argument("-f", "--fusion", type=int, default=None)
    args.add_argument("--metric", default="bleu")
    args.add_argument(
        "-l", "--logfile",
        default="logs/de_en_outroop.jsonl"
    )
    args = args.parse_args()

    model, vocab_size = me_zoo.get_model(args)

    if os.path.exists(args.logfile):
        print("Logfile already exists, refusing to continue")
        exit()

    with open(args.data, "r") as f:
        # (src, ref, hyp, conf, bleu)
        data = [
            {
                "src+hyp": sent[0] + " [SEP] " + sent[2],
                "src": sent[0],
                "ref": sent[1],
                "hyp": sent[2],
                "conf": float(sent[3]),
                "bleu": float(sent[4]) / 100,
                "chrf": float(sent[5]),
            }
            for sent in list(csv.reader(f))[:490000 + 10000]
        ]

    encoder = utils.BPEEncoder(vocab_size)
    encoder.fit([x["src+hyp"] for x in data])
    data_bpe = encoder.transform([x["src+hyp"] for x in data])
    data = [
        {"src+hyp_bpe": sent_bpe} | sent
        for sent, sent_bpe in zip(data, data_bpe)
    ]
    # the first 10k is test
    data_train = data[10000:]
    data_dev = data[:10000]

    # define logging function wrapper
    def log_step(data):
        with open(args.logfile, "a") as f:
            f.write(json.dumps(data))
            f.write("\n")
        # flushes at the end

    print(f"Training model {args.model} with fusion {args.fusion}")
    model.train_epochs(data_train, data_dev, metric=args.metric, logger=log_step)
