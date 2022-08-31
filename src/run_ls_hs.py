#!/usr/bin/env python3

import argparse
import pickle
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
import sys
sys.path.append("src")
import utils

DEVICE = utils.get_device()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d", "--data",
        default="computed/en_de_hs_f2_bleu.pkl"
    )
    args.add_argument(
        "-do", "--data-out",
        default="computed/en_de_lr_f2_bleu.pkl"
    )
    args = args.parse_args()

    print("Loading data")
    with open(args.data, "rb") as f:
        data = pickle.load(f)

    print("Preprocessing data")
    data_y = [
        1*( abs(pred_y - true_y) < 0.1)
        for pred_y, hs, true_y in data
    ]
    data_x = [
        hs
        for pred_y, hs, true_y in data
    ]

    data_dev_y = data_y[:10000]
    data_dev_x = data_x[:10000]
    data_train_y = data_y[10000:]
    data_train_x = data_x[10000:]

    print("Training & testing dummy")
    model = DummyClassifier(strategy="most_frequent")
    model.fit(data_train_x, data_train_y)
    acc_train = model.score(data_train_x, data_train_y)
    acc_dev = model.score(data_dev_x, data_dev_y)

    print(f"Accuracy train: {acc_train:.2%}")
    print(f"Accuracy dev:   {acc_dev:.2%}")

    print("Training model")
    model = LogisticRegression()
    model.fit(data_train_x, data_train_y)

    print("Running inference")
    acc_train = model.score(data_train_x, data_train_y)
    acc_dev = model.score(data_dev_x, data_dev_y)

    print(f"Accuracy train: {acc_train:.2%}")
    print(f"Accuracy dev:   {acc_dev:.2%}")

    print(list(model.predict_proba([data_x[0]])[0]))

    print("Running inference all")
    data_out = [
        (pred_y, list(model.predict_proba([hs])[0]), true_y)
        for pred_y, hs, true_y in tqdm.tqdm(data)
    ]
    print("Saving")
    print("Example:", data_out[0])
    if args.data_out is not None:
        with open(args.data_out, "wb") as f:
            pickle.dump(data_out, f)