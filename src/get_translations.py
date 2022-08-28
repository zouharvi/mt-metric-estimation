#!/usr/bin/env python3

import torch
import datasets
import tqdm
import argparse
import os
import json
# if fairseq is not imported here, it's cythoned from hub which is less robust
# possibly requires gcc >= 9.3.0?
import fairseq

DEVICE = torch.device("cuda:0")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-o", "--output", default="computed/de_en.jsonl")
    args.add_argument("--overwrite", action="store_true")
    args.add_argument("--direction", default="de-en")
    args.add_argument("-ns", "--n-start", type=int, default=0)
    args.add_argument("-ne", "--n-end", type=int, default=1000)
    args = args.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        print("The file", args.output, "already exists and you didn't --overwrite")
        print("Refusing to continue & exiting")
        exit()

    model = torch.hub.load(
        'pytorch/fairseq', f'transformer.wmt19.{args.direction}',
        checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
        tokenizer='moses', bpe='fastbpe',
        verbose=False,
    )

    if args.direction == "de-en":
        src_lang = "de"
        tgt_lang = "en"
    elif args.direction == "en-de":
        src_lang = "en"
        tgt_lang = "de"
    
    # disable dropout
    model.eval()
    model = model.to(DEVICE)

    data = datasets.load_dataset("wmt14", "de-en")["train"]

    f = open(args.output, "w")

    for sent_i, sent in enumerate(tqdm.tqdm(
        data[args.n_start*1000:args.n_end*1000]["translation"],
        total=args.n_end*1000-args.n_start*1000, miniters=100,
    )):
        sent_src = sent[src_lang]
        sent_ref = sent[tgt_lang]
        sent_src_enc = model.encode(sent_src)

        sent_tgt_enc = model.generate(sent_src_enc, nbest=5)
        sent_tgt = [(model.decode(x["tokens"]), x["score"].item()) for x in sent_tgt_enc]
        # print(sent_tgt, sent_tgt_score, sent_tgt_enc.keys())

        sent_line = {
            "src": sent_src,
            "ref": sent_ref,
            "tgts": sent_tgt,
        }
        f.write(json.dumps(sent_line, ensure_ascii=False))
        f.write("\n")

        # force flush file & tqdm
        if sent_i % 100 == 0:
            f.flush()

    f.close()
