def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

class BPEEncoder():
    def __init__(self, vocab_size):
        import bpe
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
