#!/usr/bin/env python3

import argparse
import sacrebleu
import csv
import tqdm
import evaluate

if __name__== "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", default="computed/de_en.csv")
    args.add_argument("-o", "--output", default="computed/de_en_metric.csv")
    args.add_argument("-t", "--total", type=int, default=None)
    args = args.parse_args()

    bleu_metric = sacrebleu.metrics.BLEU(effective_order=True)
    chrf_metric = sacrebleu.metrics.CHRF()
    ter_metric = sacrebleu.metrics.TER()
    meteor_metric = evaluate.load('meteor')
    comet_metric = evaluate.load('comet')

    fin = open(args.input, "r")
    fout = open(args.output, "w")
    fwriter = csv.writer(fout, quoting=csv.QUOTE_ALL)

    data_text = list(csv.reader(fin))

    print("Computing comet scores")
    evaluate.enable_progress_bar()
    comet_scores = comet_metric.compute(
        predictions=[sent[2] for sent in data_text],
        references=[sent[1] for sent in data_text],
        sources=[sent[0] for sent in data_text]
    )["scores"]
    evaluate.disable_progress_bar()

    print("Computing main loop")
    for line_i, (line, comet_score) in enumerate(tqdm.tqdm(zip(data_text, comet_scores), total=args.total)):
        sent_src = line[0]
        sent_ref = line[1]
        sent_tgt = line[2]
        conf_score = float(line[3])
        bleu_score = bleu_metric.sentence_score(hypothesis=sent_tgt, references=[sent_ref]).score
        chrf_score = chrf_metric.sentence_score(hypothesis=sent_tgt, references=[sent_ref]).score
        # this metric is quite slow
        ter_score = ter_metric.sentence_score(hypothesis=sent_tgt, references=[sent_ref]).score
        meteor_score = meteor_metric.compute(predictions=[sent_tgt], references=[sent_ref])["meteor"]

        fwriter.writerow((
            sent_src, sent_ref, sent_tgt, conf_score,
            bleu_score, chrf_score, ter_score,
            meteor_score, comet_score
        ))

        if line_i % 100 == 0:
            fout.flush()


    fin.close()
    fout.close()