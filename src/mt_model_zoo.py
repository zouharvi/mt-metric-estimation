import torch
# if fairseq is not imported here, it's cythoned from hub which is less robust
# possibly requires gcc >= 9.3.0?
import fairseq

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
            return_dict_in_generate=True,
            output_scores=True,
        )

        outs = []
        for hyp, hyp_score in zip(outputs[0], outputs[1]):
            decoded = self.tokenizer.decode(
                hyp,
                skip_special_tokens=True,
            )

            outs.append([decoded, -hyp_score.cpu().item()])

        return outs


class FairSeqWrap():
    def __init__(self, config):
        print("loading", config)
        if config.startswith("transformer.wmt18"):
            self.model = torch.hub.load(
                'pytorch/fairseq', 'transformer.wmt18.en-de',
                checkpoint_file='wmt18.model1.pt:wmt18.model2.pt:wmt18.model3.pt:wmt18.model4.pt:wmt18.model5.pt:wmt18.model6.pt',
                tokenizer='moses', bpe='subword_nmt'
            )
        elif config.startswith("transformer.wmt19"):
            self.model = torch.hub.load(
                'pytorch/fairseq', config,
                checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
                tokenizer='moses', bpe='fastbpe',
                verbose=False,
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
        sent_tgt = [(self.model.decode(x["tokens"]), x["score"].item())
                    for x in sent_tgt_enc]
        return sent_tgt


class HelsinkiWrap():
    def __init__(self, config):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"Helsinki-NLP/{config}"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            f"Helsinki-NLP/{config}"
        )
        self.model.to(DEVICE)

    def translate(self, sent_src):
        inputs = self.tokenizer(
            sent_src,
            return_tensors="pt"
        ).to(DEVICE)
        outputs = self.model.generate(
            **inputs,
            num_beams=5,
            num_return_sequences=5,
            return_dict_in_generate=True,
            output_scores=True,
        )

        outs = []
        for i in range(5):
            trans = self.tokenizer.decode(
                outputs[0][i],
                skip_special_tokens=True
            )
            score = -outputs[1][i].cpu().item()
            outs.append([trans, score])
        return outs


MODELS = {
    "t5": T5ModelWrap,
    # "w14c": lambda direction: FairSeqWrap(config=f"conv.wmt14.{direction}"),
    "w16g": lambda direction: FairSeqWrap(config=f"dynamicconv.glu.wmt16.{direction}"),
    "w17c": lambda direction: FairSeqWrap(config=f"conv.wmt17.{direction}"),
    "w16t": lambda direction: FairSeqWrap(config=f"transformer.wmt16.{direction}"),
    "w18t": lambda direction: FairSeqWrap(config=f"transformer.wmt18.{direction}"),
    "w19t": lambda direction: FairSeqWrap(config=f"transformer.wmt19.{direction}"),
    "w20t": lambda direction: FairSeqWrap(config=f"transformer.wmt20.{direction}"),
    "helsinki": lambda direction: HelsinkiWrap(config=f"opus-mt-{direction}"),
}
