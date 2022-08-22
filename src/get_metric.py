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
    args.add_argument("-t", "--total", type=int, default=None)
    args = args.parse_args()

    bleu_metric = sacrebleu.metrics.BLEU(effective_order=True)
    chrf_metric = sacrebleu.metrics.CHRF()

    fin = open(args.input, "r")
    fout = open(args.output, "w")
    fwriter = csv.writer(fout, quoting=csv.QUOTE_ALL)

    for line_i, line in enumerate(tqdm.tqdm(csv.reader(fin), total=args.total)):
        sent_src = line[0]
        sent_ref = line[1]
        sent_tgt = line[2]
        conf_score = float(line[3])
        bleu_score = bleu_metric.sentence_score(hypothesis=sent_tgt, references=[sent_ref]).score
        chrf_score = chrf_metric.sentence_score(hypothesis=sent_tgt, references=[sent_ref]).score

        fwriter.writerow((
            sent_src, sent_ref, sent_tgt, conf_score,
            bleu_score, chrf_score
        ))

        if line_i % 100 == 0:
            fout.flush()


    fin.close()
    fout.close()