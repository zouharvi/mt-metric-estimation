#!/usr/bin/env python3

import argparse
import tqdm
import evaluate
import json

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-i", "--input", default="computed/en_de_human_metric_fixed.jsonl")
    args.add_argument("-o", "--output",
                      default="computed/en_de_human_metric_brt.jsonl")
    args = args.parse_args()

    bleurt_metric = evaluate.load('bleurt', module_type="metric")

    fout = open(args.output, "w")

    with open(args.input, "r") as f:
        data_text = [json.loads(x) for x in f.readlines()]

    for line_i, sent in enumerate(tqdm.tqdm(data_text)):
        bleurt_score = bleurt_metric.compute(
            # use the first hypothesis
            predictions=[sent["tgts"][0][0]],
            references=[sent["ref"]],
        )["scores"][0]

        sent["metrics"]["bleurt"] = bleurt_score

        fout.write(json.dumps(sent, ensure_ascii=False) + "\n")

        if line_i % 100 == 0:
            fout.flush()

        if line_i % 10000 == 0:
            print(
                sent.keys(), sent["metrics"].keys(),
                sent["ref"], sent["tgts"][0][0], bleurt_score,
                sep="\n"
            )

    fout.close()
