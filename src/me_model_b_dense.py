import sys
sys.path.append("src")
import utils
import torch
import tqdm
import numpy as np

DEVICE = utils.get_device()


class MEModelBaselineDense(torch.nn.Module):
    def __init__(self, sigmoid=True, batch_size=1):
        super().__init__()

        self.batch_size = batch_size

        # TODO: BATCHING
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(6, 1),
            # TODO: add more
            # torch.nn.Linear(100, 1),
            torch.nn.Sigmoid() if sigmoid else torch.nn.Identity(),
        )

        self.loss_fn = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=10e-6)

        # move to GPU
        self.to(DEVICE)

    def forward(self, sents):
        # get gpu handle
        len_src = [len(sent["src"].split()) for sent in sents]
        len_hyp = [len(sent["hyp"].split()) for sent in sents]
        x = torch.tensor(
            [
                [
                    sent["conf"], np.exp(sent["conf"]),
                    len_src, len_hyp,
                    len_src - len_hyp,
                    len_src / len_hyp,
                ]
                for sent in sents
            ],
            dtype=torch.float32
        ).to(DEVICE)

        x = self.regressor(x)

        return x

    def eval_dev(self, data_dev):
        self.train(False)
        dev_losses = []
        dev_pred = []
        with torch.no_grad():
            for sample_i, sent in enumerate(tqdm.tqdm(data_dev)):
                # TODO: add batching
                score_pred = self.forward([sent])

                score = torch.tensor(
                    [[sent["bleu"]]], requires_grad=False
                ).to(DEVICE)
                loss = self.loss_fn(score_pred, score)

                # loss is already averaged (over 1 sample) but prediction needs to be unpacked
                dev_losses.append(loss.detach().cpu().item())
                dev_pred.append(score_pred[0].detach().cpu().item())

        return dev_losses, dev_pred

    def train_epochs(self, data_train, data_dev, epochs=30, logger=None):
        for epoch in range(epochs):
            self.train(True)

            train_losses = []
            train_pred = []
            batch = []

            for sample_i, sent in enumerate(tqdm.tqdm(data_train)):
                # this omits the last few sentences unless the total data size is divisible by batch size
                # but that's a minor oversight
                if len(batch) < self.batch_size:
                    batch.append(sent)
                    continue

                # otherwise do inference
                score_pred = self.forward(batch)

                score = torch.tensor(
                    [[sent["bleu"]] for sent in batch], requires_grad=False
                ).to(DEVICE)
                loss = self.loss_fn(score_pred, score)

                # backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.detach().cpu().item())
                # extend predictions
                train_pred += score_pred.detach().cpu().numpy().tolist()
                batch = []

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
