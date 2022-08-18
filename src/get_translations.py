#!/usr/bin/env python3

import torch
import datasets
import tqdm
import csv
import argparse
import os

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-o", "--output", default="computed/de_en.csv")
    args.add_argument("--overwrite", action="store_true")
    args = args.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        print("The file", args.output, "already exists and you didn't --overwrite")
        print("Refusing to continue & exiting")
        exit()

    model = torch.hub.load(
        'pytorch/fairseq', 'transformer.wmt19.de-en',
        checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
        tokenizer='moses', bpe='fastbpe',
        verbose=False,
    )
    # disable dropout
    model.eval()

    # data = datasets.load_dataset("wmt19", "de-en")

    data = [("Richtig supergeil", "Super cool"), ("Hallöchen", "Hello")]

    f = open(args.output, "w")
    fwriter = csv.writer(f, quoting=csv.QUOTE_ALL)

    for sent_i, sent in enumerate(tqdm.tqdm(data)):
        sent_src = sent[0]
        sent_ref = sent[1]
        sent_src_enc = model.encode(sent_src)
        # TODO: change nbest to higher numbers and see whether we can make the metric prediction better
        # with same data size
        sent_tgt_enc = model.generate(sent_src_enc, nbest=1)[0]
        sent_tgt = model.decode(sent_tgt_enc["tokens"])
        sent_tgt_score = sent_tgt_enc["score"].item()
        print(sent_tgt, sent_tgt_score, sent_tgt_enc.keys())

        fwriter.writerow((sent_src, sent_ref, sent_tgt, sent_tgt_score))

        # force flush
        if sent_i % 100 == 0:
            f.flush()

    f.close()
