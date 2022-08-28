import random
import sys
sys.path.append("src")
import utils
import torch
import tqdm
import numpy as np
from me_model_rnn import compute_extra_vector

DEVICE = utils.get_device()


class MEModelRNNMulti(torch.nn.Module):
    def __init__(
        self, vocab_size, embd_size, hidden_size, batch_size=1,
        fusion=None, dropout=0.0, num_layers=1, final_hidden_dropout=0.0,
        load_path=None, target_metrics=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.fusion = fusion

        self.target_metrics = target_metrics

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
        elif fusion == 2:
            extra_features += 10

        self.final_hidden_dropout = torch.nn.Dropout(p=final_hidden_dropout)

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(
                num_layers * hidden_size * 2 + extra_features,
                100
            ),
            torch.nn.Dropout(p=dropout),
            torch.nn.ReLU(),
            torch.nn.Linear(100, len(target_metrics)),
        )

        if load_path is not None:
            print("Loading model")
            self.load_state_dict(torch.load(load_path))

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

        x_extra = compute_extra_vector(sents, self.fusion)

        x = self.embd(x)
        x = self.encoder(x)

        # select h_n output [1][0]
        x = x[1][0]
        # bring batch to the first position and "concatenate" the rest
        x = x.transpose(1, 0).contiguous().view(local_batch_size, -1)
        # apply large dropout on the hidden state
        x = self.final_hidden_dropout(x)

        if self.fusion in {1, 2}:
            x = torch.hstack((x, x_extra))

        x = self.regressor(x)

        return x

    def eval_dev(self, data_dev):
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
                    [[sent["metrics"][metric] for metric in self.target_metrics] for sent in batch], requires_grad=False
                ).to(DEVICE)

                loss = self.loss_fn(score_pred, score)

                dev_losses.append(loss.detach().cpu().item())
                dev_pred += score_pred.detach().cpu().numpy().tolist()

                batch = []

        return dev_losses, dev_pred

    def train_epochs(self, data_train, data_dev, epochs=10, logger=None, **kwargs):
        best_dev_corr = 0

        for epoch in range(1, epochs + 1):
            self.train(True)

            train_losses = []
            train_pred = []
            batch = []

            random.shuffle(data_train)

            for sample_i, sent in enumerate(tqdm.tqdm(data_train)):
                batch.append(sent)

                if len(batch) < self.batch_size:
                    continue

                score_pred = self.forward(batch)

                score = torch.tensor(
                    [[sent["metrics"][metric] for metric in self.target_metrics] for sent in batch], requires_grad=False
                ).to(DEVICE)
                loss = self.loss_fn(score_pred, score)

                # backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.detach().cpu().item())
                train_pred += (
                    score_pred.detach().cpu().numpy().tolist()
                )

                batch = []

            # logging dev stuff
            dev_losses, dev_pred = self.eval_dev(data_dev)

            train_pred = np.array(train_pred)
            dev_pred = np.array(dev_pred)

            dev_corr = {}
            train_corr = {}

            for metric_i, metric in enumerate(self.target_metrics):
                # crop to match batch size omittance
                pred_train_metric = train_pred[:, metric_i]
                data_train_metric = [
                    sent["metrics"][metric] for sent in data_train
                ][:len(train_pred)]
                train_corr[metric] = np.corrcoef(
                    pred_train_metric, data_train_metric
                )[0, 1]

                pred_dev_metric = dev_pred[:, metric_i]
                data_dev_metric = [
                    sent["metrics"][metric] for sent in data_dev
                ][:len(dev_pred)]
                dev_corr[metric] = np.corrcoef(
                    pred_dev_metric, data_dev_metric
                )[0, 1]

            print(f"Epoch {epoch:0>5}")
            if logger is not None:
                logstep = {
                    "epoch": epoch,
                    "train_loss": np.average(train_losses),
                    "dev_loss": np.average(dev_losses),
                    "train_corr": train_corr,
                    "dev_corr": dev_corr,
                }
                logger(logstep)

            if kwargs.get("save_path") is not None and kwargs.get("save_metric") is not None:
                if abs(dev_corr[kwargs["save_metric"]]) > abs(best_dev_corr):
                    best_dev_corr = dev_corr[kwargs["save_metric"]]
                    print(
                        f"Saving model because new dev_corr of {kwargs['save_metric']} is {best_dev_corr:.3f}"
                    )
                    torch.save(self.state_dict(), kwargs["save_path"])
