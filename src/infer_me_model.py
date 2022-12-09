#!/usr/bin/env python3

import sys
sys.path.append("src")
import me_zoo

import json
import argparse
import os
import tqdm
import utils
import pickle

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d", "--data",
        default="computed/en_de_human_test.jsonl"
    )
    args.add_argument(
        "-o", "--output",
        default="computed/en_de_human_test.pred"
    )
    args.add_argument("-m", "--model", default="1hd75b10lin")
    args.add_argument("-f", "--fusion", type=int, default=None)
    args.add_argument("--shuffle-train", type=int, default=None)
    args.add_argument(
        "-sb", "--save-bpe", default=None,
        help="Store BPE model (path)"
    )
    args.add_argument(
        "-lb", "--load-bpe", default=None,
        help="Load BPE model (path)"
    )
    # should have None default otherwise we always just fine-tune
    args.add_argument("-mp", "--model-load-path", default=None)
    args.add_argument(
        "-l", "--logfile",
        default="logs/de_en_outroop.jsonl"
    )
    args = args.parse_args()

    model, vocab_size = me_zoo.get_model(args)

    if os.path.exists(args.logfile) and not args.save_bpe_only:
        print("Logfile already exists, refusing to continue")
        exit()

    with open(args.data, "r") as f:
        data = [json.loads(x) for x in f.readlines()]

    extractor = lambda sent: sent["src"] + " [SEP] " + sent["tgts"][0][0]

    data = [
        sent | {
            "text": extractor(sent),
            "hyp": sent["tgts"][0][0],
        }
        for sent in data
    ]

    # skip for models that don't use BPE
    if not args.model in {"b", "comet", "mbert"}:
        if args.load_bpe is None:
            encoder = utils.BPEEncoder(vocab_size)
            encoder.fit([x["text"] for x in data])
        else:
            with open(args.load_bpe, "rb") as f:
                encoder = pickle.load(f)

        data_bpe = encoder.transform([x["text"] for x in data])
        data = [
            {"text_bpe": sent_bpe} | sent
            for sent, sent_bpe in zip(data, data_bpe)
        ]
        if args.save_bpe is not None:
            print("Saving BPE model to", args.save_bpe)
            with open(args.save_bpe, "wb") as f:
                pickle.dump(encoder, f)
            if args.save_bpe_only:
                exit()
    
    print(f"Inferring model {args.model} with fusion {args.fusion}")
    with open(args.output, "w") as f:    
        for line in tqdm.tqdm(data):
            score = model.forward([line])[0].item()
            f.write(f"{score}\n")