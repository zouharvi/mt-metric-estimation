#!/usr/bin/env python3

import argparse
import tqdm
import json

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-i", "--input",
    )
    args.add_argument(
        "-o", "--output",
    )
    args = args.parse_args()

    with open(args.input, "r") as f:
        data_text = [json.loads(x) for x in f.readlines()]

    fout = open(args.output, "w")

    for line_i, sent in enumerate(tqdm.tqdm(data_text)):
        for f in ["h1_hx_bleu_avg", "h1_hx_bleu_var", "hx_hx_bleu_avg", "hx_hx_bleu_var"]:
            sent[f] = sent["metrics"][f]

        fout.write(json.dumps(sent, ensure_ascii=False) + "\n")

        if line_i % 100 == 0:
            fout.flush()

    fout.close()
