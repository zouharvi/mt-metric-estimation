import sys
sys.path.append("src")
import utils
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tqdm

DEVICE = utils.get_device()


class MEModelBaseline():
    def __init__(self):
        pass

    def train_epochs(self, data_train, data_dev, metric="bleu", metric_dev=None, logger=None):
        y_train = [sent["metrics"][metric] for sent in data_train]
        y_dev = [sent["metrics"][metric_dev] for sent in data_dev]

        model = LinearRegression()

        def featurizer(sent):
            len_src = len(sent["src"].split())
            len_hyp = len(sent["hyp"].split())
            return (
                len_src, len_hyp,
                len_src - len_hyp,
                len_src / len_hyp,
                sent["conf"],
                np.exp(sent["conf"]),
            )

        x_train = [
            featurizer(sent) for sent in data_train
        ]
        x_dev = [
            featurizer(sent) for sent in data_dev
        ]

        model.fit(x_train, y_train)

        ypred_train = model.predict(x_train)
        ypred_dev = model.predict(x_dev)

        train_mse = mean_squared_error(y_train, ypred_train)
        dev_mse = mean_squared_error(y_dev, ypred_dev)

        logstep = {
            "model": "lr_multi",
            "train_mse": train_mse,
            "dev_mse": dev_mse,
            "train_corr": np.corrcoef(ypred_train, y_train)[0, 1],
            "dev_corr": np.corrcoef(ypred_dev, y_dev)[0, 1],
        }
        logger(logstep)

        for features_exp in tqdm.tqdm([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
            try:
                tfidf = TfidfVectorizer(max_features=2**features_exp)
                model = LinearRegression()

                x_train = tfidf.fit_transform(
                    [sent["src+hyp"] for sent in data_train])
                x_dev = tfidf.fit_transform([sent["src+hyp"] for sent in data_dev])
                model.fit(x_train, y_train)

                ypred_train = model.predict(x_train)
                ypred_dev = model.predict(x_dev)

                train_mse = mean_squared_error(y_train, ypred_train)
                dev_mse = mean_squared_error(y_dev, ypred_dev)

                logstep = {
                    "model": f"tfidf_lr_2^{features_exp}",
                    "train_mse": train_mse,
                    "dev_mse": dev_mse,
                    "train_corr": np.corrcoef(ypred_train, y_train)[0, 1],
                    "dev_corr": np.corrcoef(ypred_dev, y_dev)[0, 1],
                }
                logger(logstep)
            except:
                pass
                # TFIDF too large mismatch

        ypred_train = [
            sent["src+hyp"].strip().count(" ")
            for sent in data_train
        ]
        ypred_dev = [
            sent["src+hyp"].strip().count(" ")
            for sent in data_dev
        ]
