import sys
sys.path.append("src")
import utils
import torch
import bpe
import tqdm
import numpy as np

DEVICE = utils.get_device()


class Encoder():
    def __init__(self, vocab_size):
        self.bpe = bpe.Encoder(vocab_size=vocab_size)

    def fit(self, data):
        assert type(data) == list
        self.bpe.fit(data)

    def transform(self, data):
        if type(data) == list:
            return self.bpe.transform(data)
        elif type(data) is str:
            return self.bpe.transform([data])[0]
        else:
            raise Exception("Unknown input data type")

    def tokenize(self, data):
        if type(data) == list:
            return self.bpe.tokenize(data)
        elif type(data) is str:
            return self.bpe.tokenize([data])[0]
        else:
            raise Exception("Unknown input data type")

    def decode(self, data):
        # not used in this project
        pass


class MEModel1(torch.nn.Module):
    def __init__(self, vocab_size, embd_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size

        self.embd = torch.nn.Linear(vocab_size, embd_size)

        # this encoder models concatenated (src, hyp)
        # TODO: add attention
        self.encoder = torch.nn.LSTM(
            input_size=embd_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, 100),
            torch.nn.Linear(100, 1),
            # TODO: remove me
            torch.nn.Sigmoid(),
        )

        self.loss_fn = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=10e-6)

        # move to GPU
        self.to(DEVICE)

    def forward(self, x):
        x = self.embd(x)
        # print("a", x.shape)

        # add fake batch
        x = x.reshape((1, *x.shape))
        # print("b", x.shape)
        # TODO: add batching
        x = self.encoder(x)
        # select h_n output [1][0], take first in batch [:,0,:]
        x = x[1][0][:, 0, :]
        # concatenate forward and backward runs
        # print("c", x.shape)
        x = torch.hstack((x[0], x[1]))
        # print("d", x.shape)
        x = self.regressor(x)
        # print("e", x.shape)
        return x

    def eval_dev(self, data_dev):
        self.train(False)
        dev_losses = []
        dev_pred = []
        with torch.no_grad():
            for sample_i, sent in enumerate(tqdm.tqdm(data_dev)):
                # TODO: add batching

                # get gpu handle
                sent_bpe = torch.tensor(sent["src+hyp_bpe"])
                sent_bpe = torch.nn.functional.one_hot(
                    sent_bpe, num_classes=self.vocab_size).float().to(DEVICE)
                score_pred = self.forward(sent_bpe)

                score = torch.tensor(
                    [sent["bleu"]], requires_grad=False).to(DEVICE)
                loss = self.loss_fn(score_pred, score)

                dev_losses.append(loss.detach().cpu().item())
                dev_pred.append(score_pred.detach().cpu().item())

        return dev_losses, dev_pred

    def train_epochs(self, data_train, data_dev, epochs=50, logger=None):
        for epoch in range(1, epochs + 1):
            self.train(True)

            train_losses = []
            train_pred = []

            for sample_i, sent in enumerate(tqdm.tqdm(data_train)):
                # TODO: add batching

                # get gpu handle
                sent_bpe = torch.tensor(sent["src+hyp_bpe"])
                sent_bpe = torch.nn.functional.one_hot(
                    sent_bpe, num_classes=self.vocab_size).float().to(DEVICE)
                score_pred = self.forward(sent_bpe)

                score = torch.tensor(
                    [sent["bleu"]], requires_grad=False).to(DEVICE)
                loss = self.loss_fn(score_pred, score)

                # backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.detach().cpu().item())
                train_pred.append(score_pred.detach().cpu().item())

            # logging dev stuff
            dev_losses, dev_pred = self.eval_dev(data_dev)
            data_dev_score = [sent["bleu"] for sent in data_dev]
            data_train_score = [sent["bleu"] for sent in data_train]

            print(f"Epoch {epoch:0>5}")
            if logger is not None:
                logstep = {
                    "epoch": epoch,
                    "train_loss": np.average(train_losses),
                    "dev_loss": np.average(dev_losses),
                    "train_corr": np.corrcoef(train_pred, data_train_score)[0, 1],
                    "dev_corr": np.corrcoef(dev_pred, data_dev_score)[0, 1],
                }
                logger(logstep)
