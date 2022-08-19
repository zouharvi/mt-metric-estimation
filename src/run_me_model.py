#!/usr/bin/env python3

import argparse
import json
from me_model_1 import MEModel1, Encoder
import csv
import os

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-d", "--data", default="computed/de_en_metric.csv")
    args.add_argument("-m", "--model", default="1")
    args.add_argument("-l", "--logfile",
                      default="computed/de_en_outroop.jsonl")
    args.add_argument("-vs", "--vocab-size", type=int, default=8192)
    args = args.parse_args()

    if args.model == "1":
        model = MEModel1(args.vocab_size, 512, 128)
    else:
        raise Exception("Unknown model")

    if os.path.exists(args.logfile):
        print("Logfile already exists, refusing to continue")
        exit()

    with open(args.data, "r") as f:
        # (src, ref, hyp, conf, bleu)
        data = [
            {
                "src+hyp": sent[0] + " | " + sent[2],
                "src": sent[0],
                "ref": sent[1],
                "hyp": sent[2],
                "conf": float(sent[3]),
                "bleu": float(sent[4]) / 100,
            }
            for sent in list(csv.reader(f))
        ][:40000+1000]

    encoder = Encoder(args.vocab_size)
    encoder.fit([x["src+hyp"] for x in data])
    data_bpe = encoder.transform([x["src+hyp"] for x in data])
    data = [
        {"src+hyp_bpe": sent_bpe} | sent
        for sent, sent_bpe in zip(data, data_bpe)
    ]
    data_train = data[1000:]
    data_dev = data[:1000]

    def log_step(data):
        with open(args.logfile, "a") as f:
            f.write(json.dumps(data))
            f.write("\n")
        # flushes at the end

    model.train_epochs(data_train, data_dev, logger=log_step)
