#!/usr/bin/env python3

from collections import defaultdict
import matplotlib.pyplot as plt
import sys
sys.path.append("src")
from figures import fig_utils
import argparse
import json

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-l", "--logfiles", nargs="+",
    )
    args = args.parse_args()

    for fn in args.logfiles:
        with open(fn, "r") as f:
            data = [
                (line_i, json.loads(line))
                for line_i, line in enumerate(f.readlines())
            ]
            model_best_epoch = max(
                data, key=lambda x: x[1]["dev_corr"]
            )
            print(fn, f"{model_best_epoch[0]+1}/{len(data)}")
            print(model_best_epoch[1])