#!/usr/bin/env python3

import argparse
import tqdm
import json
from mt_model_zoo import MODELS

# TODO: this script shares most of the functionality of get_translations.py

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", default="computed/en_de.jsonl")
    args.add_argument("-o", "--output", default="computed/en_de_t5.jsonl")
    args.add_argument("-dn", "--data-n", type=int, default=10000)
    args.add_argument("-dir", "--direction", default="en-de")
    args.add_argument("-m", "--model", default=None)
    args = args.parse_args()

    model = MODELS[args.model](args.direction)

    with open(args.input, "r") as f:
        data = [json.loads(x) for x in f.readlines()[:args.data_n]]

    fout = open(args.output, "w")

    print("Computing main loop")
    for line_i, sent in enumerate(tqdm.tqdm(data)):
        # translate 5 new hypotheses
        sent["tgts"] = model.translate(sent["src"])

        # the top one hypothesis is always the one with highest score but make sure
        sent["tgts"] = sorted(sent["tgts"], key=lambda x: x[1], reverse=True)

        # get first hypothesis
        sent_tgt = sent["tgts"][0][0]

        fout.write(json.dumps(sent, ensure_ascii=False) + "\n")

        if line_i % 100 == 0:
            fout.flush()

    fout.close()
