#!/usr/bin/env python3

import torch
import datasets
import tqdm
import csv
import argparse
import os
import sys
# if fairseq is not imported here, it's cythoned from hub which is less robust
# possibly requires gcc >= 9.3.0?
import fairseq

DEVICE = torch.device("cuda:0")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-o", "--output", default="computed/de_en.csv")
    args.add_argument("--overwrite", action="store_true")
    args.add_argument("-t", "--total", type=int, default=1000000)
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
    model = model.to(DEVICE)

    data = datasets.load_dataset("wmt14", "de-en")["train"]["translation"]

    f = open(args.output, "w")
    fwriter = csv.writer(f, quoting=csv.QUOTE_ALL)

    for sent_i, sent in enumerate(tqdm.tqdm(
        data[:args.total],
        total=args.total, miniters=100,
    )):
        sent_src = sent["de"]
        sent_ref = sent["en"]
        sent_src_enc = model.encode(sent_src)
        # TODO: change nbest to higher numbers and see whether we can make the metric prediction
        # better with same data size
        sent_tgt_enc = model.generate(sent_src_enc, nbest=1)[0]
        sent_tgt = model.decode(sent_tgt_enc["tokens"])
        sent_tgt_score = sent_tgt_enc["score"].item()
        # print(sent_tgt, sent_tgt_score, sent_tgt_enc.keys())

        fwriter.writerow((sent_src, sent_ref, sent_tgt, sent_tgt_score))

        # force flush file & tqdm
        if sent_i % 100 == 0:
            f.flush()
            sys.stderr.flush()

    f.close()
