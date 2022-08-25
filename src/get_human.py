#!/usr/bin/env python3

import torch
import datasets
import tqdm
import argparse
import os
import json
import csv
# if fairseq is not imported here, it's cythoned from hub which is less robust
# possibly requires gcc >= 9.3.0?
import fairseq

from fairseq.sequence_scorer import SequenceScorer

DEVICE = torch.device("cuda:0")

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", default="computed/en_de_human.csv")
    args.add_argument("--overwrite", action="store_true")
    args.add_argument("--direction", default="en-de")
    args.add_argument("-o", "--output", default="computed/en_de_human.jsonl")
    args = args.parse_args()

    fin = open(args.input, "r")
    data = csv.DictReader(fin)

    unique_sents = set()
    total_sents = 0

    if os.path.exists(args.output) and not args.overwrite:
        print("The file", args.output, "already exists and you didn't --overwrite")
        print("Refusing to continue & exiting")
        exit()

    # model = torch.hub.load(
    #     'pytorch/fairseq', f'transformer.wmt19.{args.direction}',
    #     checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
    #     tokenizer='moses', bpe='fastbpe',
    #     verbose=False,
    # )

    if args.direction == "de-en":
        src_lang = "de"
        tgt_lang = "en"
    elif args.direction == "en-de":
        src_lang = "en"
        tgt_lang = "de"

    # disable dropout
    # model.eval()
    # model = model.to(DEVICE)

    fout = open(args.output, "w")

    for sent in tqdm.tqdm(data):
        # header of a second file concatenated
        if sent["score"] == "score":
            continue

        total_sents += 1
        unique_sents.add(sent["src"] + " ||| " + sent["ref"])

        # TODO: get confidence

        # sent_src_enc = model.encode("hey how are you")
        # sent_tgt_enc = model.encode("hallo wie gehts dir")

        # scorer = SequenceScorer(model.tgt_dict)
        # scorer.generate(model, {"net_input": sent_src_enc, "target": sent_tgt_enc})

        # def restrictor(batch_id, input_ids):
        #     with open("~/test", "w") as f:
        #         f.write("xx")
        #     print(batch_id, input_ids)
        # model.generate(sent_src_enc, prefix_allowed_tokens_fn=restrictor)
        # add_module', 'apply', 'apply_bpe', 'bfloat16', 'binarize', 'buffers', 'children', 'cpu', 'cuda', 'decode', 'detokenize', 'double', 'encode', 'eval', 'extra_repr', 'float', 'forward', 'generate', 'get_buffer', 'get_extra_state', 'get_parameter', 'get_submodule', 'half', 'ipu', 'load_state_dict', 'models', 'modules', 'named_buffers', 'named_children', 'named_modules', 'named_parameters', 'parameters', 'register_backward_hook', 'register_buffer', 'register_forward_hook', 'register_forward_pre_hook', 'register_full_backward_hook', 'register_load_state_dict_post_hook', 'register_module', 'register_parameter', 'remove_bpe', 'requires_grad_', 'sample', 'score', 'set_extra_state', 'share_memory', 'state_dict', 'string', 'to', 'to_empty', 'tokenize', 'train', 'translate', 'type', 'xpu', 'zero_grad']

        print(sent["score"], sent["zscore"])
        sent_line = {
            "src": sent["src"],
            "ref": sent["ref"],
            "tgts": [[sent["mt"], 0]],
            "score": float(sent["score"]),
            "zscore": float(sent["zscore"]),
        }
        fout.write(json.dumps(sent_line, ensure_ascii=False))
        fout.write("\n")

    print(f"Total {total_sents}, unique {len(unique_sents)}")
    fout.close()