import sys
sys.path.append("src")
import utils
import torch
import tqdm
import numpy as np

DEVICE = utils.get_device()


class MEModelRNN(torch.nn.Module):
    def __init__(
        self, vocab_size, embd_size, hidden_size, batch_size=1,
        fusion=None, sigmoid=True, relu=False, dropout=0.0, num_layers=1, final_hidden_dropout=0.0, sigmoid_scale=1.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.fusion = fusion
        assert sigmoid_scale >= 1.0
        self.sigmoid_scale = sigmoid_scale
        self.sigmoid_offset = (sigmoid_scale - 1) / 2

        self.embd = torch.nn.Linear(vocab_size, embd_size)

        # this encoder models concatenated (src, hyp)
        # TODO: add attention
        self.encoder = torch.nn.LSTM(
            input_size=embd_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
            num_layers=num_layers,
            # only aplied when num_layers > 1
            dropout=0.2,
        )

        extra_features = 0
        if fusion == 1:
            extra_features += 6

        self.final_hidden_dropout = torch.nn.Dropout(p=final_hidden_dropout)

        # TODO: deeper model
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(num_layers * hidden_size *
                            2 + extra_features, 100),
            torch.nn.Dropout(p=dropout),
            torch.nn.ReLU() if relu else torch.nn.Identity(),
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid() if sigmoid else torch.nn.Identity(),
        )

        self.loss_fn = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=10e-6)

        # move to GPU
        self.to(DEVICE)

    def forward(self, sents):
        local_batch_size = len(sents)

        # get gpu handle
        x = [
            torch.nn.functional.one_hot(
                torch.tensor(sent["src+hyp_bpe"]), num_classes=self.vocab_size
            ).float().to(DEVICE) for sent in sents
        ]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)

        len_src = [len(sent["src"].split()) for sent in sents]
        len_hyp = [len(sent["hyp"].split()) for sent in sents]
        x_extra = torch.tensor(
            [
                [
                    sent["conf"], np.exp(sent["conf"]),
                    len_src[sent_i], len_hyp[sent_i],
                    len_src[sent_i] - len_hyp[sent_i],
                    len_src[sent_i] / len_hyp[sent_i],
                ]
                for sent_i, sent in enumerate(sents)
            ],
            dtype=torch.float32
        ).to(DEVICE)

        x = self.embd(x)
        x = self.encoder(x)

        # select h_n output [1][0]
        x = x[1][0]
        # bring batch to the first position and "concatenate" the rest
        x = x.transpose(1, 0).contiguous().view(local_batch_size, -1)
        # apply large dropout on the hidden state
        x = self.final_hidden_dropout(x)

        if self.fusion == 1:
            x = torch.hstack((x, x_extra))

        x = self.regressor(x)

        # multiply final sigmoid output and center
        x = x * self.sigmoid_scale - self.sigmoid_offset

        return x

    def eval_dev(self, data_dev, metric):
        self.train(False)
        dev_losses = []
        dev_pred = []
        batch = []

        with torch.no_grad():
            for sample_i, sent in enumerate(tqdm.tqdm(data_dev)):
                batch.append(sent)

                if len(batch) < self.batch_size:
                    continue

                score_pred = self.forward(batch)

                score = torch.tensor(
                    [[sent[metric]] for sent in batch], requires_grad=False
                ).to(DEVICE)
                loss = self.loss_fn(score_pred, score)

                dev_losses.append(loss.detach().cpu().item())
                dev_pred += score_pred.reshape(-1).detach().cpu().numpy().tolist()

                batch = []

        return dev_losses, dev_pred

    def train_epochs(self, data_train, data_dev, metric="bleu", epochs=10, logger=None):
        for epoch in range(1, epochs + 1):
            self.train(True)

            train_losses = []
            train_pred = []
            batch = []

            # TODO shuffle

            for sample_i, sent in enumerate(tqdm.tqdm(data_train)):
                batch.append(sent)

                if len(batch) < self.batch_size:
                    continue

                score_pred = self.forward(batch)

                score = torch.tensor(
                    [[sent[metric]] for sent in batch], requires_grad=False
                ).to(DEVICE)
                loss = self.loss_fn(score_pred, score)

                # backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.detach().cpu().item())
                train_pred += score_pred.reshape(-1).detach(
                ).cpu().numpy().tolist()

                batch = []

            # logging dev stuff
            dev_losses, dev_pred = self.eval_dev(data_dev, metric=metric)
            # crop to match batch size omittance
            data_train_score = [
                sent[metric] for sent in data_train
            ][:len(train_pred)]
            data_dev_score = [
                sent[metric] for sent in data_dev
            ][:len(dev_pred)]

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
