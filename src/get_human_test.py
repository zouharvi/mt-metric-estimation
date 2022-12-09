#!/usr/bin/env python3

import torch
import tqdm
import argparse
import os
import json
import csv
import itertools
from transformers import T5Tokenizer, T5ForConditionalGeneration

DEVICE = torch.device("cuda:0")


def calculate_logprob(input_text, output_text, model, tokenizer):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)
    output_ids = tokenizer.encode(output_text, return_tensors="pt").to(DEVICE)
    outputs = model(input_ids, labels=output_ids)
    return -outputs[0].item()


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument(
        "-i1", "--input-src", default="/home/vilda/Downloads/en_de_test.src",
    )
    args.add_argument(
        "-i2", "--input-tgt", default="/home/vilda/Downloads/en_de_test.tgt",
    )
    args.add_argument("--overwrite", action="store_true")
    args.add_argument("--direction", default="en-de")
    args.add_argument("-o", "--output", default="computed/en_de_human_test.jsonl")
    args = args.parse_args()

    f1 = open(args.input_src, "r").readlines()
    f2 = open(args.input_tgt, "r").readlines()
    data = [(x.rstrip("\n"), y.rstrip("\n")) for x, y in zip(f1, f2)]

    unique_sents = set()
    total_sents = 0

    if os.path.exists(args.output) and not args.overwrite:
        print("The file", args.output, "already exists and you didn't --overwrite")
        print("Refusing to continue & exiting")
        exit()

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

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

    for sent_src, sent_tgt in tqdm.tqdm(data):
        total_sents += 1
        unique_sents.add(sent_src)

        conf = calculate_logprob(
            task_prefix + sent_src, sent_tgt, model, tokenizer
        )

        sent_line = {
            "src": sent_src,
            # "ref": sent["ref"],
            "tgts": [[sent_tgt, conf]],
            # "score": float(sent["score"]),
            # "zscore": float(sent["zscore"]),
        }
        fout.write(json.dumps(sent_line, ensure_ascii=False))
        fout.write("\n")

    print(f"Total {total_sents}, unique {len(unique_sents)}")
    fout.close()
