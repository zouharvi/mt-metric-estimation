import sys
sys.path.append("src")
import utils
import torch
import tqdm
import numpy as np

DEVICE = utils.get_device()

class MEModelRNN(torch.nn.Module):
    def __init__(self, vocab_size, embd_size, hidden_size, fusion=None, sigmoid=True, relu=False, dropout=0.0, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.fusion = fusion

        self.embd = torch.nn.Linear(vocab_size, embd_size)

        # this encoder models concatenated (src, hyp)
        # TODO: add attention
        self.encoder = torch.nn.LSTM(
            input_size=embd_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
            num_layers=num_layers,
        )

        extra_features = 0
        if fusion == 1:
            extra_features += 6

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2 + extra_features, 100),
            torch.nn.Dropout(p=dropout),
            torch.nn.ReLU() if relu else torch.nn.Identity(),
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid() if sigmoid else torch.nn.Identity(),
        )

        self.loss_fn = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=10e-6)

        # move to GPU
        self.to(DEVICE)

    def forward(self, sent):
        # get gpu handle
        x = torch.tensor(sent["src+hyp_bpe"])
        x = torch.nn.functional.one_hot(
            x, num_classes=self.vocab_size
        ).float().to(DEVICE)

        len_src = len(sent["src"].split())
        len_hyp = len(sent["hyp"].split())
        x_extra = torch.tensor([
            sent["conf"], np.exp(sent["conf"]),
            len_src, len_hyp,
            len_src - len_hyp,
            len_src / len_hyp,
        ]
        ).to(DEVICE)

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
        if self.fusion == 1:
            x = torch.hstack((x, x_extra))
        # print("e", x.shape)
        x = self.regressor(x)
        # print("f", x.shape)
        return x

    def eval_dev(self, data_dev):
        self.train(False)
        dev_losses = []
        dev_pred = []
        with torch.no_grad():
            for sample_i, sent in enumerate(tqdm.tqdm(data_dev)):
                # TODO: add batching
                score_pred = self.forward(sent)

                score = torch.tensor(
                    [sent["bleu"]], requires_grad=False
                ).to(DEVICE)
                loss = self.loss_fn(score_pred, score)

                dev_losses.append(loss.detach().cpu().item())
                dev_pred.append(score_pred.detach().cpu().item())

        return dev_losses, dev_pred

    def train_epochs(self, data_train, data_dev, epochs=10, logger=None):
        for epoch in range(1, epochs + 1):
            self.train(True)

            train_losses = []
            train_pred = []

            for sample_i, sent in enumerate(tqdm.tqdm(data_train)):
                # TODO: add batching
                score_pred = self.forward(sent)

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
