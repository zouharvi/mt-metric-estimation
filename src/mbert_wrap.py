import sys
sys.path.append("src")
import utils
from transformers import BertTokenizer, BertModel

DEVICE = utils.get_device()

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-multilingual-cased'
)
model = BertModel.from_pretrained(
    'bert-base-multilingual-cased'
).to(DEVICE)

def get_mbert_representations(sents):
    x = tokenizer.batch_encode_plus(
        [sent["text"] for sent in sents],
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        padding="longest",
        # we pad to longest but don't go over maximum total tokens
        truncation=True,
        return_attention_mask=False,
        return_tensors='pt',
    ).to(DEVICE)

    x = model(**x)

    x = x.last_hidden_state[:,0]
    
    return x