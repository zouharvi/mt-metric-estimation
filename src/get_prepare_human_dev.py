#!/usr/bin/env python3

import argparse
from collections import defaultdict
import tqdm
import json
import numpy as np

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", default="computed/en_de_human_metric.jsonl")
    args.add_argument("-o", "--output", default="computed/en_de_human_metric_fixed.jsonl")
    args.add_argument("-t", "--total", type=int, default=None)
    args = args.parse_args()

    fout = open(args.output, "w")

    with open(args.input, "r") as f:
        data_in = [json.loads(x) for x in f.readlines()]

    data_main = defaultdict(list)

    for line_i, sent in enumerate(data_in):
        data_main[sent["src"]].append(sent)

    # becomes sorted by src
    for vs in data_main.values():
        print(len(vs))
        for sent in vs:
            fout.write(json.dumps(sent, ensure_ascii=False) + "\n")

    fout.close()
