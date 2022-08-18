#!/usr/bin/env python3

import argparse
import sacrebleu
import csv
import tqdm

if __name__== "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--metric", default="bleu")
    args.add_argument("-i", "--input", default="computed/de_en.csv")
    args.add_argument("-o", "--output", default="computed/de_en_metric.csv")
    args = args.parse_args()

    if args.metric == "bleu":
        metric = sacrebleu.metrics.BLEU(effective_order=True)
    else:
        raise Exception(f"Unknown metric {args.metric}")

    fin = open(args.input, "r")
    fout = open(args.output, "w")
    fwriter = csv.writer(fout, quoting=csv.QUOTE_ALL)

    for line_i, line in enumerate(tqdm.tqdm(csv.reader(fin))):
        sent_src = line[0]
        sent_ref = line[1]
        sent_tgt = line[2]
        conf_score = float(line[3])
        metric_score = metric.sentence_score(hypothesis=sent_tgt, references=[sent_ref]).score

        print(line, metric_score)
        fwriter.writerow((sent_src, sent_ref, sent_tgt, conf_score, metric_score))

        if line_i % 100 == 0:
            fout.flush()


    fin.close()
    fout.close()