#!/usr/bin/env python3

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("src")
from figures import fig_utils
import argparse
import json

# scp euler:/cluster/work/sachan/vilem/mt-metric-estimation/computed/jeren/*.jsonl computed/jeren/

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args = args.parse_args()

    data = {}

    plt.figure(figsize=(5, 3))

    # DIRECTIONS = ["en_cs", "fr_en", "en_de"]
    DIRECTIONS = ["fr_en", "en_de"]
    for direction in DIRECTIONS:
        with open(f"computed/jeren/{direction}_predicted.jsonl", "r") as f:
            data_local = [json.loads(line) for line in f.readlines()]
            data[direction] = [
                (sent["metrics"]["ter"], sent["metrics_pred"]["ter"])
                for sent in data_local
            ]
            if direction == "en_de":
                data[direction] = [(x[0], x[1]/100) for x in data[direction]]

            data[direction] = [(x[0], x[1]) for x in data[direction] if x[1] >= -1 and x[1] <= 2]

            corrcoef = np.corrcoef(
                [x[0] for x in data[direction]],
                [x[1] for x in data[direction]],
            )[0,1]
            print(direction, data[direction][1], corrcoef)
            

    for direction, data in data.items():
        print("plotting", direction)
        plt.scatter(
            [x[0]*100 for x in data],
            [x[1]*100 for x in data],
            label=direction,
            alpha=0.3,
            s=10,
        )

    plt.xlim(0, 100)        
    plt.ylim(0, 100)        
    plt.ylabel("True TER")
    plt.xlabel("Predicted TER")

    plt.legend(
        ncol=1
        # , bbox_to_anchor=(0.45, 1.27), loc="upper center"
    )
    plt.tight_layout(pad=0.1)
    # rect=(-0.015, 0, 1.03, 1.0), pad=0.1)
    plt.savefig("figures/metric_scatter.pdf")
    plt.show()
