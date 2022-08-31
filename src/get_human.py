#!/usr/bin/env python3

import torch
import datasets
import tqdm
import argparse
import os
import json
import csv
import itertools
from transformers import T5Tokenizer, T5ForConditionalGeneration

DEVICE = torch.device("cuda:0")


def calculate_logprob(input_text, output_text, model, tokenizer):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output_ids = tokenizer.encode(output_text, return_tensors="pt")
    outputs = model(input_ids, labels=output_ids)
    return -outputs[0].item()

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument(
        "-i", "--input", nargs="+",
        default=[
            "computed/en_de_human_0.csv",
            "computed/en_de_human_0.csv",
        ]
    )
    args.add_argument("--overwrite", action="store_true")
    args.add_argument("--direction", default="en-de")
    args.add_argument("-o", "--output", default="computed/en_de_human.jsonl")
    args = args.parse_args()

    fins = [open(f, "r") for f in args.input]
    data = itertools.chain(*[csv.DictReader(f) for f in fins])

    unique_sents = set()
    total_sents = 0

    if os.path.exists(args.output) and not args.overwrite:
        print("The file", args.output, "already exists and you didn't --overwrite")
        print("Refusing to continue & exiting")
        exit()

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    # TODO this may not work
    model.eval()
    model.to(DEVICE)

    if args.direction == "en-de":
        task_prefix = "translate English to German: "
    else:
        task_prefix = "translate German to English: "

    if args.direction == "de-en":
        src_lang = "de"
        tgt_lang = "en"
    elif args.direction == "en-de":
        src_lang = "en"
        tgt_lang = "de"

    fout = open(args.output, "w")

    for sent in tqdm.tqdm(data):
        # header of a second file concatenated
        if sent["score"] == "score":
            continue

        total_sents += 1
        unique_sents.add(sent["src"] + " ||| " + sent["ref"])

        conf = calculate_logprob(
            task_prefix + sent["src"], sent["mt"], model, tokenizer
        )
        print(conf)

        sent_line = {
            "src": sent["src"],
            "ref": sent["ref"],
            "tgts": [[sent["mt"], conf]],
            "score": float(sent["score"]),
            "zscore": float(sent["zscore"]),
        }
        fout.write(json.dumps(sent_line, ensure_ascii=False))
        fout.write("\n")

    print(f"Total {total_sents}, unique {len(unique_sents)}")
    fout.close()
