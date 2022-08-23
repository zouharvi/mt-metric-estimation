#!/usr/bin/env python3

import argparse
import sacrebleu
import csv
import tqdm

if __name__== "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", default="computed/de_en_metric_all.csv")
    args.add_argument("-o", "--output", default="computed/de_en_metric_all_fixed.csv")
    args = args.parse_args()

    ter_metric = sacrebleu.metrics.TER()

    fin = open(args.input, "r")
    fout = open(args.output, "w")
    fwriter = csv.writer(fout, quoting=csv.QUOTE_ALL)

    data_text = list(csv.reader(fin))

    print("Computing main loop")
    for line_i, line in enumerate(tqdm.tqdm(data_text)):
        sent_src = line[0]
        sent_ref = line[1]
        sent_tgt = line[2]
        conf_score = float(line[3])
        bleu_score = float(line[4])
        chrf_score = float(line[5])
        # ter_score = float(line[6])
        meteor_score = float(line[7])
        comet_score = float(line[8])

        # this metric is quite slow
        ter_score = ter_metric.sentence_score(hypothesis=sent_tgt, references=[sent_ref]).score

        fwriter.writerow((
            sent_src, sent_ref, sent_tgt, conf_score,
            bleu_score, chrf_score, ter_score,
            meteor_score, comet_score
        ))

        if line_i % 100 == 0:
            fout.flush()


    fin.close()
    fout.close()