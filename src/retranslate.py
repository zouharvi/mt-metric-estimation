#!/usr/bin/env python3

import argparse
import tqdm
import json
import torch
import fairseq
from fairseq.models.fconv import FConvModel

DEVICE = torch.device("cuda:0")

class T5ModelWrap():
    def __init__(self, direction):
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        if direction == "en-de":
            self.task_prefix = "translate English to German: "
        else:
            self.task_prefix = "translate German to English: "
        
        self.model.eval()
        self.model.to(DEVICE)

    def translate(self, sent_src):
        input_ids = self.tokenizer.encode(
            sent_src,
            return_tensors="pt",
            max_length=512,
            truncation="longest_first",
        ).to(DEVICE)
        outputs = self.model.generate(
            input_ids,
            num_return_sequences=5,
            num_beams=5,
            max_length=512,
        )

        outs = []
        for hyp in outputs:
            decoded = self.tokenizer.decode(
                hyp,
                skip_special_tokens=False,
            )
            # we could extract the probability in the generation step but there were some issues
            # this duplicates the computation but does not matter much because it's just for eval
            decoded = decoded.replace("<pad>", "").strip().rstrip("</s>")
            output_ids = self.tokenizer.encode(decoded, return_tensors="pt")
            output_ids = output_ids.to(DEVICE)
            output_score = -self.model(input_ids, labels=output_ids)[0].item()

            outs.append([decoded, output_score])

        return outs

class FairSeqWrap():
    def __init__(self, config):
        print("loading", config)
        if config.startswith("transformer.wmt18"):
            torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt18.en-de',
                checkpoint_file='wmt18.model1.pt:wmt18.model2.pt:wmt18.model3.pt:wmt18.model4.pt:wmt18.model5.pt:wmt18.model6.pt',
                tokenizer='moses', bpe='subword_nmt'
            )
        else:
            self.model = torch.hub.load(
                'pytorch/fairseq', config,
                tokenizer='moses', bpe='subword_nmt',
                verbose=False,
            )

        self.model.eval()
        self.model.to(DEVICE)

    def translate(self, sent_src):
        sent_src_enc = self.model.encode(sent_src)
        sent_tgt_enc = self.model.generate(sent_src_enc, nbest=5)
        sent_tgt = [(self.model.decode(x["tokens"]), x["score"].item()) for x in sent_tgt_enc]
        return sent_tgt

MODELS = {
    "t5": T5ModelWrap,
    # "w14c": lambda direction: FairSeqWrap(config=f"conv.wmt14.{direction}"),
    "w16g": lambda direction: FairSeqWrap(config=f"dynamicconv.glu.wmt16.{direction}"),
    "w17c": lambda direction: FairSeqWrap(config=f"conv.wmt17.{direction}"),
    "w16t": lambda direction: FairSeqWrap(config=f"transformer.wmt16.{direction}"),
    "w18t": lambda direction: FairSeqWrap(config=f"transformer.wmt18.{direction}"),
}

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-i", "--input", default="computed/en_de.jsonl")
    args.add_argument("-o", "--output", default="computed/en_de_t5.jsonl")
    args.add_argument("-dn", "--data-n", type=int, default=10000)
    args.add_argument("-dir", "--direction", default="en-de")
    args.add_argument("-m", "--model", default=None)
    args = args.parse_args()

    model = MODELS[args.model](args.direction)

    with open(args.input, "r") as f:
        data = [json.loads(x) for x in f.readlines()[:args.data_n]]

    fout = open(args.output, "w")

    print("Computing main loop")
    for line_i, sent in enumerate(tqdm.tqdm(data)):
        # translate 5 new hypotheses
        sent["tgts"] = model.translate(sent["src"])

        # the top one hypothesis is always the one with highest score but make sure
        sent["tgts"] = sorted(sent["tgts"], key=lambda x: x[1], reverse=True)

        # get first hypothesis
        sent_tgt = sent["tgts"][0][0]

        fout.write(json.dumps(sent, ensure_ascii=False) + "\n")

        if line_i % 100 == 0:
            fout.flush()

    fout.close()
