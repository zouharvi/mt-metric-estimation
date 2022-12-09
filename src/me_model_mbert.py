import random
import sys
sys.path.append("src")
import utils
import torch
import tqdm
import numpy as np
from transformers import BertTokenizer, BertModel

DEVICE = utils.get_device()


class MEModelMBERT(torch.nn.Module):
    def __init__(
        self, batch_size=1,
        load_path=None, **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-multilingual-cased'
        )
        self.model = BertModel.from_pretrained(
            'bert-base-multilingual-cased'
        )

        # TODO: deeper model
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(768, 100),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1),
        )

        if load_path is not None:
            print("Loading model", load_path)
            self.load_state_dict(torch.load(load_path))

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=10e-6)

        # move to GPU
        self.to(DEVICE)

    def forward(self, sents):
        local_batch_size = len(sents)

        x = self.tokenizer.batch_encode_plus(
            [sent["text"] for sent in sents],
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            padding="longest",
            # we pad to longest but don't go over maximum total tokens
            truncation=True,
            return_attention_mask=False,
            return_tensors='pt',
        ).to(DEVICE)

        x = self.model(**x)

        x = x.last_hidden_state[:,0]

        x = self.regressor(x)

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
                    [[sent["metrics"][metric]] for sent in batch], requires_grad=False
                ).to(DEVICE)
                loss = self.loss_fn(score_pred, score)

                dev_losses.append(loss.detach().cpu().item())
                dev_pred += score_pred.reshape(-1).detach().cpu().numpy().tolist()

                batch = []

        return dev_losses, dev_pred

    def train_epochs(
        self, data_train, data_dev,
        metric="bleu", metric_dev=None,
        epochs=10, logger=None,
        **kwargs
    ):
        best_dev_corr = 0

        if metric_dev == None:
            metric_dev = metric

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
                    [[sent["metrics"][metric]] for sent in batch], requires_grad=False
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
            dev_losses, dev_pred = self.eval_dev(
                data_dev, metric=metric_dev,
            )
            # crop to match batch size omittance
            data_train_score = [
                sent["metrics"][metric] for sent in data_train
            ][:len(train_pred)]
            data_dev_score = [
                sent["metrics"][metric_dev] for sent in data_dev
            ][:len(dev_pred)]

            print(f"Epoch {epoch:0>5}")
            if logger is not None:
                dev_corr = np.corrcoef(dev_pred, data_dev_score)[0, 1]
                logstep = {
                    "epoch": epoch,
                    "train_loss": np.average(train_losses),
                    "dev_loss": np.average(dev_losses),
                    "train_corr": np.corrcoef(train_pred, data_train_score)[0, 1],
                    "dev_corr": dev_corr,
                }
                logger(logstep)
            if "save_path" in kwargs is not None:
                if abs(dev_corr) > abs(best_dev_corr):
                    best_dev_corr = dev_corr
                    print(
                        f"Saving model because new dev_corr is {dev_corr:.3f}"
                    )
                    torch.save(self.state_dict(), kwargs["save_path"])
