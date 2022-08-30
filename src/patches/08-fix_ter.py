#!/usr/bin/env python3

import argparse
import tqdm
import json

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-i", "--input",
        default="computed/en_de_human_metric_brt.jsonl"
    )
    args.add_argument(
        "-o", "--output",
        default="computed/en_de_human_metric_ft.jsonl"
    )
    args = args.parse_args()

    fout = open(args.output, "w")

    with open(args.input, "r") as f:
        data_text = [json.loads(x) for x in f.readlines()]

    for line_i, sent in enumerate(tqdm.tqdm(data_text)):
        sent["metrics"]["ter"] = sent["metrics"]["ter"] / 100

        fout.write(json.dumps(sent, ensure_ascii=False) + "\n")

        if line_i % 100 == 0:
            fout.flush()

    fout.close()
