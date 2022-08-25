#!/usr/bin/env python3

import argparse
import sacrebleu
import tqdm
import evaluate
import json
import numpy as np

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", default="computed/de_en.jsonl")
    args.add_argument("-o", "--output", default="computed/de_en_metric.jsonl")
    args.add_argument("-t", "--total", type=int, default=None)
    args = args.parse_args()

    bleu_metric = sacrebleu.metrics.BLEU(effective_order=True)
    chrf_metric = sacrebleu.metrics.CHRF()
    ter_metric = sacrebleu.metrics.TER()
    meteor_metric = evaluate.load('meteor')
    comet_metric = evaluate.load('comet')

    fout = open(args.output, "w")

    with open(args.input, "r") as f:
        data_text = [json.loads(x) for x in f.readlines()]
    

    if "score" in data_text[0]:
        human_file = True
        print("Detected human scores file")
    else:
        human_file = False

    # the progress bar here is incorrect but maybe it just shows batches instead of sentences?
    print("Computing comet scores")
    comet_scores = comet_metric.compute(
        # use the first hypothesis
        predictions=[sent["tgts"][0][0] for sent in data_text],
        references=[sent["ref"] for sent in data_text],
        sources=[sent["src"] for sent in data_text],
        progress_bar=True,
    )["scores"]

    print("Computing main loop")
    for line_i, (sent, comet_score) in enumerate(tqdm.tqdm(zip(data_text, comet_scores), total=len(comet_scores))):
        # get first hypothesis
        sent_tgt = sent["tgts"][0][0]

        # the top one hypothesis is always the one with highest score
        # best_tgt = sorted(sent["tgts"], key=lambda x: x[1], reverse=True)[0]
        # assert sent_tgt == best_tgt[0]

        # assign decoder confidence
        sent["conf"] = sent["tgts"][0][1]
        sent["conf_exp"] = np.exp(sent["tgts"][0][1])

        if not human_file:
            sent["conf_var"] = np.var([x[1] for x in sent["tgts"]])
            sent["conf_exp_var"] = np.var([np.exp(x[1]) for x in sent["tgts"]])
            h1_hx_bleu = [
                bleu_metric.sentence_score(
                    hypothesis=sent_tgt, references=[x[0]]
                ).score / 100
                for x in sent["tgts"][1:]
            ]
            sent["h1_hx_bleu_avg"] = np.average(h1_hx_bleu)
            sent["h1_hx_bleu_var"] = np.var(h1_hx_bleu)
            hx_hx_bleu = [
                bleu_metric.sentence_score(
                    hypothesis=x[0], references=[y[0]]
                ).score / 100
                for x_i, x in enumerate(sent["tgts"])
                for y_i, y in enumerate(sent["tgts"])
                if x_i != y_i
            ]
            sent["hx_hx_bleu_avg"] = np.average(hx_hx_bleu)
            sent["hx_hx_bleu_var"] = np.var(hx_hx_bleu)


        bleu_score = bleu_metric.sentence_score(
            hypothesis=sent_tgt, references=[sent["ref"]]
        ).score / 100
        chrf_score = chrf_metric.sentence_score(
            hypothesis=sent_tgt, references=[sent["ref"]]
        ).score / 100
        ter_score = ter_metric.sentence_score(
            hypothesis=sent_tgt, references=[sent["ref"]]
        ).score
        meteor_score = meteor_metric.compute(
            predictions=[sent_tgt], references=[sent["ref"]]
        )["meteor"]

        sent["metrics"] = {
            "bleu": bleu_score,
            "chrf": chrf_score,
            "ter": ter_score,
            "meteor": meteor_score,
            "comet": comet_score,
        }

        if human_file:
            sent["metrics"]["score"] = sent.pop("score")
            sent["metrics"]["zscore"] = sent.pop("zscore")
            
        fout.write(json.dumps(sent, ensure_ascii=False) + "\n")

        if line_i % 100 == 0:
            fout.flush()

    fout.close()
