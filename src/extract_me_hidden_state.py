#!/usr/bin/env python3

import json
import argparse
import pickle
import torch
import tqdm
import sys
sys.path.append("src")
import utils
import me_zoo

DEVICE = utils.get_device()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d", "--data",
        default="computed/en_de_metric_ft.jsonl"
    )
    args.add_argument("-m", "--model", default="1hd75b10lin")
    args.add_argument("-f", "--fusion", type=int, default=None)
    args.add_argument("-dn", "--data-n", type=int, default=None)
    args.add_argument("--no-dropout", action="store_true")
    args.add_argument(
        "-lb", "--load-bpe", default="models/bpe_news_500k_h1.pkl",
        help="Load BPE model (path)"
    )
    # should have None default otherwise we always just fine-tune
    args.add_argument("-mp", "--model-load-path", default=None)
    args.add_argument("--metric", default="bleu")
    args.add_argument(
        "-do", "--data-out",
        default="computed/en_de_hs_f2_bleu.pkl"
    )
    args = args.parse_args()

    if args.model == "1hd75b10lind20" and args.no_dropout:
        print("Mismatch between --model and --no-dropout")

    model, vocab_size = me_zoo.get_model(args)

    with open(args.data, "r") as f:
        data = [json.loads(x) for x in f.readlines()][:args.data_n]

    # (src, ref, hyp, conf, bleu)
    data = [
        sent | {
            "text": sent["src"] + " [SEP] " + sent["tgts"][0][0],
            "hyp": sent["tgts"][0][0],
        }
        for sent in data
    ]

    with open(args.load_bpe, "rb") as f:
        encoder = pickle.load(f)

    data_bpe = encoder.transform([x["text"] for x in data])
    data = [
        {"text_bpe": sent_bpe} | sent
        for sent, sent_bpe in zip(data, data_bpe)
    ]

    if not args.no_dropout:
        # multiply 50 times
        data = [sent for sent in data for _ in range(50)]
    else:
        # turn dropout off
        model.eval()

    print("Running inference")
    data_pred = []
    data_hs = []
    data_y = []
    batch = []
    with torch.no_grad():
        for sample_i, sent in enumerate(tqdm.tqdm(data)):
            batch.append(sent)

            if len(batch) < model.batch_size:
                continue

            score_pred, score_hs = model.forward(batch, output_hs=True)

            score = torch.tensor(
                [[sent["metrics"][args.metric]] for sent in batch], requires_grad=False
            ).to(DEVICE)

            data_pred += score_pred.reshape(-1).detach().cpu().numpy().tolist()
            data_y += score.reshape(-1).detach().cpu().numpy().tolist()
            data_hs += score_hs.detach().cpu().numpy().tolist()
            batch = []

    with open(args.data_out, "wb") as f:
        pickle.dump(list(zip(data_pred, data_hs, data_y)), file=f)