#!/usr/bin/env python3

import argparse
import tqdm
import json
import numpy as np
import copy 

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", default="computed/en_de.jsonl")
    args.add_argument("-o", "--output", default="computed/en_de_h.jsonl")
    args.add_argument("-hn", "--hypothesis-n", type=int, default=1)
    args = args.parse_args()

    fout = open(args.output, "w")

    with open(args.input, "r") as f:
        data_in = [json.loads(x) for x in f.readlines()]
    
    data_out = []

    print("Computing main loop")
    for line_i, sent in enumerate(tqdm.tqdm(data_in)):
        for hyp_i, hyp in enumerate(sent["tgts"][:args.hypothesis_n]):
            sent_new = copy.deepcopy(sent)
            h1 = sent_new["tgts"].pop(hyp_i)
            # put a specific hypothesis at the beginning
            sent_new["tgts"] = [h1] + sent_new["tgts"]
            data_out.append(sent_new)
            fout.write(json.dumps(sent_new, ensure_ascii=False) + "\n")

        if line_i % 100 == 0:
            fout.flush()

    fout.close()
