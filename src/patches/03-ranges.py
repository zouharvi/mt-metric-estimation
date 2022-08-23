#!/usr/bin/env python3

import argparse
import csv
import tqdm

if __name__== "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", default="computed/de_en_metric_all_fixed.csv")
    args = args.parse_args()


    fin = open(args.input, "r")
    data_text = list(csv.reader(fin))
    fin.close()

    values = [[] for _ in range(6)]

    for line_i, line in enumerate(tqdm.tqdm(data_text)):
        for f_i in range(3, 8+1):
            values[f_i-3].append(float(line[f_i]))

    for f_i in range(3, 8+1):
        print(f"{f_i}: ({min(values[f_i-3])}, {max(values[f_i-3])})")