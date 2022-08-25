import sys
sys.path.append("src")
import utils
import numpy as np
import evaluate
import tqdm

DEVICE = utils.get_device()


class MEModelComet():
    def __init__(self):
        # TODO: try wmt21-comet-qe-da
        self.comet_metric = evaluate.load("comet", config_name='wmt21-comet-qe-da')

    def train_epochs(self, data_train, data_dev, metric, logger=None):
        data_dev_pred = self.comet_metric.compute(
            predictions=[sent["hyp"] for sent in data_dev],
            sources=[sent["src"] for sent in data_dev],
            references=["" for sent in data_dev],
            progress_bar=True,
        )["scores"]
        data_dev_y = [sent["metrics"][metric] for sent in data_dev]
        corr = np.corrcoef(data_dev_y, data_dev_pred)[0,1]
        print(f"Corrcoef: {corr:.10f}")