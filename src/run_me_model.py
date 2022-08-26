#!/usr/bin/env python3

import sys
sys.path.append("src")
import utils
import argparse
import json
import os
import me_zoo

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-dt", "--data-train", default="computed/en_de_human_metric.jsonl")
    args.add_argument("-dd", "--data-dev", default=None)
    args.add_argument("-m", "--model", default="1")
    args.add_argument("-f", "--fusion", type=int, default=None)
    args.add_argument("-dn", "--dev-n", type=int, default=None)
    args.add_argument("-tn", "--train-n", type=int, default=None)
    args.add_argument("--metric", default="bleu")
    args.add_argument("--metric-dev", default="zscore")
    args.add_argument(
        "-l", "--logfile",
        default="logs/de_en_outroop.jsonl"
    )
    args = args.parse_args()

    model, vocab_size = me_zoo.get_model(args)

    if os.path.exists(args.logfile):
        print("Logfile already exists, refusing to continue")
        exit()

    with open(args.data_train, "r") as f:
        data_train = [json.loads(x) for x in f.readlines()]
        data = data_train
    if args.data_dev is not None:
        with open(args.data_dev, "r") as f:
            data_dev = [json.loads(x) for x in f.readlines()]

        if args.dev_n is not None:
            data_dev = data_dev[:args.dev_n]

        if "human" in args.data_dev and args.dev_n != 1000:
            print("You're using the human data but your dev-n is not 1k as described in the paper")
            exit()
            
        # data_dev is first
        data = data_dev + data_train
        args.dev_n = len(data_dev)
    else:
        if "human" in args.data_train and args.dev_n != 1000:
            print("You're using the human data but your dev-n is not 1k as described in the paper")
            exit()
        if args.dev_n is None:
            print("Unkown dev size specified")
            exit()
        
    if args.train_n is None:
        args.train_n = len(data_train)

    # (src, ref, hyp, conf, bleu)
    data = [
        sent | {
            "src+hyp": sent["src"] + " [SEP] " + sent["tgts"][0][0],
            "hyp": sent["tgts"][0][0],
        }
        for sent in data
    ]

    if args.model.startswith("1"):
        encoder = utils.BPEEncoder(vocab_size)
        encoder.fit([x["src+hyp"] for x in data])
        data_bpe = encoder.transform([x["src+hyp"] for x in data])
        data = [
            {"src+hyp_bpe": sent_bpe} | sent
            for sent, sent_bpe in zip(data, data_bpe)
        ]

    # the first 1k/10k is test
    data_train = data[args.dev_n:args.train_n+args.dev_n]
    data_dev = data[:args.dev_n]

    # define logging function wrapper
    def log_step(data):
        with open(args.logfile, "a") as f:
            f.write(json.dumps(data))
            f.write("\n")
        # flushes at the end

    print(f"Training model {args.model} with fusion {args.fusion}")
    model.train_epochs(data_train, data_dev, metric=args.metric, metric_dev=args.metric_dev, logger=log_step)
