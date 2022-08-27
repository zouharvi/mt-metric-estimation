import sys
sys.path.append("src")
import utils
import numpy as np
import evaluate
import tqdm

DEVICE = utils.get_device()


class MEModelComet():
    def __init__(self):
        # wmt21-comet-qe-mqm works better than wmt21-comet-qe-da
        self.model_name = "wmt21-comet-qe-mqm"
        self.comet_metric = evaluate.load("comet", config_name=self.model_name)

    def train_epochs(self, data_train, data_dev, logger=None, **kwargs):
        data_dev_pred = self.comet_metric.compute(
            predictions=[sent["hyp"] for sent in data_dev],
            sources=[sent["src"] for sent in data_dev],
            references=["" for sent in data_dev],
            progress_bar=True,
        )["scores"]

        print("Disregarding main & dev metric and running against all")

        METRICS = list(data_dev[0]["metrics"].keys())
        for metric in METRICS:
            data_dev_y = [sent["metrics"][metric] for sent in data_dev]
            corr = np.corrcoef(data_dev_y, data_dev_pred)[0,1]
            print(f"{metric:>10}-comet: {corr:.3f}")
            logger({"model": self.model_name, "metric": metric, "dev_corr": corr})
