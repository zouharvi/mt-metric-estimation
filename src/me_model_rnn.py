import random
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
        fusion=None, sigmoid=True, relu=False, dropout=0.0, num_layers=1, final_hidden_dropout=0.0, sigmoid_scale=1.0,
        load_path=None,
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
        elif fusion == 2:
            extra_features += 10
        elif fusion == 3:
            extra_features += 10 + 768
            # load bert only if needed
            from mbert_wrap import get_mbert_representations
            self.get_mbert_representations = get_mbert_representations

        self.final_hidden_dropout = torch.nn.Dropout(p=final_hidden_dropout)

        # TODO: deeper model
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(
                num_layers * hidden_size * 2 + extra_features,
                100
            ),
            torch.nn.Dropout(p=dropout),
            torch.nn.ReLU() if relu else torch.nn.Identity(),
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid() if sigmoid else torch.nn.Identity(),
        )

        if load_path is not None:
            print("Loading model")
            self.load_state_dict(torch.load(load_path))

        self.loss_fn = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=10e-6)

        # move to GPU
        self.to(DEVICE)

    def forward(self, sents, output_hs=False):
        local_batch_size = len(sents)

        # get gpu handle
        x = [
            torch.nn.functional.one_hot(
                torch.tensor(sent["text_bpe"]), num_classes=self.vocab_size
            ).float().to(DEVICE) for sent in sents
        ]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)

        if self.fusion != 0:
            x_extra = self.compute_extra_vector(sents, self.fusion)

        x = self.embd(x)
        x = self.encoder(x)

        # select h_n output [1][0]
        x = x[1][0]
        # bring batch to the first position and "concatenate" the rest
        x = x.transpose(1, 0).contiguous().view(local_batch_size, -1)
        # apply large dropout on the hidden state
        x = self.final_hidden_dropout(x)

        if self.fusion in {1, 2, 3}:
            x = torch.hstack((x, x_extra))

        if output_hs:
            for layer_i, layer in enumerate(self.regressor):
                if layer_i == 3:
                    hs = x
                x = layer(x)
        else:
            # faster
            x = self.regressor(x)
        
        # multiply final sigmoid output and center
        x = x * self.sigmoid_scale - self.sigmoid_offset

        if output_hs:
            return (x, hs)
        else:
            return x

    def eval_dev(self, data_dev, metric, scale_metric=1):
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
                    [[sent["metrics"][metric] * scale_metric] for sent in batch], requires_grad=False
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
        scale_metric = kwargs["scale_metric"]

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
                    [[sent["metrics"][metric] * scale_metric] for sent in batch], requires_grad=False
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
                scale_metric=scale_metric
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
                        f"Saving model because new dev_corr is {dev_corr:.3f}")
                    torch.save(self.state_dict(), kwargs["save_path"])


    def compute_extra_vector(self, sents, fusion):
        len_src = [len(sent["src"].split()) for sent in sents]
        len_hyp = [len(sent["hyp"].split()) for sent in sents]
        if fusion == 1:
            x_extra_1 = torch.tensor(
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
            return x_extra_1
        elif fusion == 2:
            x_extra_1 = torch.tensor(
                [
                    [
                        sent["conf"], np.exp(sent["conf"]),
                        len_src[sent_i], len_hyp[sent_i],
                        len_src[sent_i] - len_hyp[sent_i],
                        len_src[sent_i] /
                        len_hyp[sent_i] if len_hyp[sent_i] != 0 else 0,
                        sent["h1_hx_bleu_avg"], sent["h1_hx_bleu_var"],
                        sent["hx_hx_bleu_avg"], sent["hx_hx_bleu_var"],
                    ]
                    for sent_i, sent in enumerate(sents)
                ],
                dtype=torch.float32
            ).to(DEVICE)
            return x_extra_1
        elif fusion == 3:
            x_extra_1 = torch.tensor(
                [
                    [
                        sent["conf"], np.exp(sent["conf"]),
                        len_src[sent_i], len_hyp[sent_i],
                        len_src[sent_i] - len_hyp[sent_i],
                        len_src[sent_i] /
                        len_hyp[sent_i] if len_hyp[sent_i] != 0 else 0,
                        sent["h1_hx_bleu_avg"], sent["h1_hx_bleu_var"],
                        sent["hx_hx_bleu_avg"], sent["hx_hx_bleu_var"],
                    ]
                    for sent_i, sent in enumerate(sents)
                ],
                dtype=torch.float32
            ).to(DEVICE)
            x_extra_2 = self.get_mbert_representations(sents)
            x_extra = torch.hstack((x_extra_1, x_extra_2))
            return x_extra