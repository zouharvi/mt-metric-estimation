#!/usr/bin/env python3

import os
import json
import argparse
import random
import utils
import pickle
import sys
sys.path.append("src")
import me_zoo

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-dt", "--data-train",
        default="computed/en_de_human_metric_ft.jsonl"
    )
    args.add_argument("-dd", "--data-dev", default=None)
    args.add_argument("-m", "--model", default="1hd75b10lin")
    args.add_argument("-f", "--fusion", type=int, default=None)
    args.add_argument("--epochs", type=int, default=10)
    args.add_argument("-dn", "--dev-n", type=int, default=None)
    args.add_argument("-tn", "--train-n", type=int, default=None)
    args.add_argument("--scale-metric", type=int, default=1)
    args.add_argument("--shuffle-train", type=int, default=None)
    args.add_argument(
        "-sb", "--save-bpe", default=None,
        help="Store BPE model (path)"
    )
    args.add_argument(
        "-lb", "--load-bpe", default=None,
        help="Load BPE model (path)"
    )
    # should have None default otherwise we always just fine-tune
    args.add_argument("-mp", "--model-load-path", default=None)
    args.add_argument(
        "-hn", "--hypothesis-n", type=int, default=1,
        help="Has to be specified so that dev & train set is correct (when get_expand_hyp is used)"
    )
    args.add_argument("--metric", default="bleu")
    args.add_argument("--metric-dev", default=None)
    args.add_argument("--save-metric", default="bleu")
    args.add_argument("--text-feature", default="src+hyp")
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
            print(
                "You're using the human data but your dev-n is not 1k as described in the paper")
            exit()

        # data_dev is first
        data = data_dev + data_train
        args.dev_n = len(data_dev)
    else:
        if "human" in args.data_train and args.dev_n != 1000:
            print(
                "You're using the human data but your dev-n is not 1k as described in the paper")
            exit()
        if args.dev_n is None:
            print("Unkown dev size specified")
            exit()

    if args.train_n is None:
        args.train_n = len(data_train)

    if args.text_feature == "src+hyp":
        extractor = lambda sent: sent["src"] + " [SEP] " + sent["tgts"][0][0]
    elif args.text_feature == "src":
        extractor = lambda sent: sent["src"]
    elif args.text_feature == "hyp":
        extractor = lambda sent: sent["tgts"][0][0]

    # (src, ref, hyp, conf, bleu)
    data = [
        sent | {
            "text": extractor(sent),
            "hyp": sent["tgts"][0][0],
        }
        for sent in data
    ]

    # skip for models that don't use BPE
    if not args.model in {"b", "comet", "mbert"}:
        if args.load_bpe is None:
            encoder = utils.BPEEncoder(vocab_size)
            encoder.fit([x["text"] for x in data])
        else:
            with open(args.load_bpe, "rb") as f:
                encoder = pickle.load(f)

        data_bpe = encoder.transform([x["text"] for x in data])
        data = [
            {"text_bpe": sent_bpe} | sent
            for sent, sent_bpe in zip(data, data_bpe)
        ]
        if args.save_bpe is not None:
            print("Saving BPE model to", args.save_bpe)
            with open(args.save_bpe, "wb") as f:
                pickle.dump(encoder, f)
    
    # the first 1k/10k is test
    data_train = data[args.dev_n * args.hypothesis_n:]
    # sample randomly
    if args.shuffle_train is not None:
        random.seed(args.shuffle_train)
        random.shuffle(data_train)
    data_train = data_train[:args.train_n * args.hypothesis_n]
    # skip every n-th in the development part (take only the first)
    data_dev = data[:args.dev_n * args.hypothesis_n:args.hypothesis_n]

    # define logging function wrapper
    def log_step(data):
        with open(args.logfile, "a") as f:
            f.write(json.dumps(data))
            f.write("\n")
        # flushes at the end

    # set default evaluation metric
    if args.metric_dev is None:
        args.metric_dev = args.metric
    print(f"Training model {args.model} with fusion {args.fusion}")
    model.train_epochs(
        data_train, data_dev,
        epochs=args.epochs,
        # disregarded for multi model
        metric=args.metric, metric_dev=args.metric_dev,
        logger=log_step,
        # used only by the multi model
        save_metric=args.save_metric,
        save_path=(
            args.logfile.replace("logs/", "models/").replace(".jsonl", ".pt")
        ),
        scale_metric=args.scale_metric,
    )
