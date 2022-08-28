#!/usr/bin/env python3

import os
import sys
sys.path.append("src")
import argparse
import json


def multi_score(dev_corrs):
    return sum([
        abs(v)
        for v in dev_corrs.values()
    ])


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-l", "--logfiles", nargs="+",
    )
    args = args.parse_args()

    for fn in args.logfiles:
        if not os.path.isfile(fn):
            continue
        with open(fn, "r") as f:
            data = [
                [line_i, json.loads(line)]
                for line_i, line in enumerate(f.readlines())
            ]
            if type(data[0][1]["dev_corr"]) == dict:
                # take maximum sum of absolute correlations
                model_best_epoch = max(
                    data,
                    key=lambda x: multi_score(x[1]["dev_corr"])
                )
                multi_score_val = multi_score(model_best_epoch[1]["dev_corr"])
                model_best_epoch[1]["dev_corr_total"] = multi_score_val
                model_best_epoch[1]["dev_corr_avg"] = (
                    multi_score_val / len(model_best_epoch[1]["dev_corr"])
                )
            else:
                model_best_epoch = max(
                    data, key=lambda x: abs(x[1]["dev_corr"])
                )
            print()
            print(fn, f"{model_best_epoch[0]+1}/{len(data)}")
            print(model_best_epoch[1])

    print()