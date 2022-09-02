#!/usr/bin/env python3

import torch
import datasets
import tqdm
import argparse
import os
import json
from mt_model_zoo import MODELS


DEVICE = torch.device("cuda:0")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-o", "--output", default="computed/en_de.jsonl")
    args.add_argument("--overwrite", action="store_true")
    args.add_argument("--direction", default="en-de")
    args.add_argument("-ns", "--n-start", type=int, default=0)
    args.add_argument("-ne", "--n-end", type=int, default=1000)
    args.add_argument("-m", "--model", default="w19t")
    args = args.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        print("The file", args.output, "already exists and you didn't --overwrite")
        print("Refusing to continue & exiting")
        exit()

    model = MODELS[args.model](args.direction)

    print("Testing translate capabilities")
    print("hello?", model.translate("Hello"))

    langs = args.direction.split("-")
    src_lang = langs[0]
    tgt_lang = langs[1]
    if args.direction in {"de-en", "en-de"}:
        dataset_name = "de-en"
    elif args.direction in {"ru-en", "en-ru"}:
        dataset_name = "ru-en"
    elif args.direction in {"fr-en", "en-fr"}:
        dataset_name = "fr-en"

    data = datasets.load_dataset("wmt14", dataset_name)["train"]
    print("Total available", len(data))

    f = open(args.output, "w")

    for sent_i, sent in enumerate(tqdm.tqdm(
        data[args.n_start*1000:args.n_end*1000]["translation"],
        total=args.n_end*1000-args.n_start*1000, miniters=100,
    )):
        sent_src = sent[src_lang]
        sent_ref = sent[tgt_lang]

        # translate 5 new hypotheses
        sent_tgt = model.translate(sent_src)

        # the top one hypothesis is always the one with highest score but make sure
        sent_tgt = sorted(sent_tgt, key=lambda x: x[1], reverse=True)

        sent_line = {
            "src": sent_src,
            "ref": sent_ref,
            "tgts": sent_tgt,
        }
        f.write(json.dumps(sent_line, ensure_ascii=False))
        f.write("\n")

        # force flush file & tqdm
        if sent_i % 100 == 0:
            f.flush()

    f.close()
