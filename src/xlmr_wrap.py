import sys
sys.path.append("src")
import utils
from transformers import AutoTokenizer, AutoModelForMaskedLM

DEVICE = utils.get_device()

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base").to(DEVICE)

def get_xlmr_representations(sents):
    x = tokenizer.batch_encode_plus(
        [sent["text"] for sent in sents],
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        padding="longest",
        # we pad to longest but don't go over maximum total tokens
        truncation=True,
        return_attention_mask=False,
        return_tensors='pt',
    ).to(DEVICE)

    x = model(**x, output_hidden_states=True)

    # take just the embedding layer [0]
    # [:, 0] accross all batches and last element
    x = x.hidden_states[1][:, 0]

    return x
