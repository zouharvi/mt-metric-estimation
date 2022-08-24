#!/usr/bin/env python3

import argparse
import json

if __name__== "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", default="computed/en_de_metric.jsonl")
    args.add_argument("-o", "--output", default="computed/en_de_metric_fixed.jsonl")
    args = args.parse_args()

    with open(args.input, "r") as f:
        data = [json.loads(x) for x in f.readlines()]

    for line in data:
        line["h1_hx_bleu_avg"] = line.pop("h1_x_bleu_avg")
        line["h1_hx_bleu_var"] = line.pop("h1_x_bleu_var")
        line["hx_hx_bleu_avg"] = line.pop("hx_x_bleu_avg")
        line["hx_hx_bleu_var"] = line.pop("hx_x_bleu_var")

    with open(args.output, "w") as f:
        f.writelines([json.dumps(x, ensure_ascii=False)+"\n" for x in data])