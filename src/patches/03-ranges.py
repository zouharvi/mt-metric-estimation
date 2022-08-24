#!/usr/bin/env python3

import argparse
import json
import tqdm
from collections import defaultdict

if __name__== "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", default="computed/en_de_metric.jsonl")
    args = args.parse_args()

    with open(args.input, "r") as f:
        data = [json.loads(x) for x in f.readlines()]

    values = defaultdict(list)

    for line_i, line in enumerate(tqdm.tqdm(data)):
        for m_k, m_v in line["metrics"].items():
            values[m_k].append(m_v)

    for m_k, m_v in values.items():
        print(f"{m_k}: ({min(m_v)}, {max(m_v)})")