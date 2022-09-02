#!/usr/bin/env python3

import torch
import datasets
import tqdm
import argparse
import os
import json
from mt_model_zoo import MODELS


DEVICE = torch.device("cuda:0")

DATASET_MAP = {
    "en-de": "wmt14",
    "de-en": "wmt14",
    "en-fr": "wmt14",
    "fr-en": "wmt14",
    "en-ru": "wmt14",
    "ru-en": "wmt14",
    "en-zh": "ccmatrix",
    "zh-en": "ccmatrix",
    "en-hi": "ccmatrix",
    "hi-en": "ccmatrix",
    "pl-de": "opus_paracrawl",
    "de-pl": "opus_paracrawl",
}

def get_dataset(direction):
    name = DATASET_MAP[direction]
    if name == "wmt14":
        if direction in {"de-en", "en-de"}:
            return datasets.load_dataset("wmt14", "de-en")["train"]
        elif direction in {"ru-en", "en-ru"}:
            return datasets.load_dataset("wmt14", "ru-en")["train"]
        elif direction in {"fr-en", "en-fr"}:
            return datasets.load_dataset("wmt14", "fr-en")["train"]
    elif name == "ccmatrix":
        if direction in {"zh-en", "en-zh"}:
            return datasets.load_dataset("yhavinga/ccmatrix", lang1="en", lang2="zh")["train"]
        elif direction in {"hi-en", "hi-zh"}:
            return datasets.load_dataset("yhavinga/ccmatrix", lang1="en", lang2="hi")["train"]
    elif name == "opus_paracrawl":
        if direction in {"pl-de", "de-pl"}:
            return datasets.load_dataset("opus_paracrawl", "de-pl")["train"]

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-o", "--output", default="computed/en_de.jsonl")
    args.add_argument("--overwrite", action="store_true")
    args.add_argument("-ns", "--n-start", type=int, default=0)
    args.add_argument("-ne", "--n-end", type=int, default=1000)
    args.add_argument("-m", "--model", default="w19t")
    args.add_argument("--direction", default="en-de")
    args.add_argument("--dry-dataset", action="store_true", help="Only download the model & data")
    args.add_argument("--dry-model", action="store_true", help="Only download the model")
    args = args.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        print("The file", args.output, "already exists and you didn't --overwrite")
        print("Refusing to continue & exiting")
        exit()

    model = MODELS[args.model](args.direction)

    print("Testing translate capabilities")
    print("hello?", model.translate("Hello"))

    if args.dry_model:
        exit()

    langs = args.direction.split("-")
    src_lang = langs[0]
    tgt_lang = langs[1]
    
    data = get_dataset(args.dataset, args.direction)
    print("Total available", len(data))

    if args.dry_dataset:
        exit()

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
