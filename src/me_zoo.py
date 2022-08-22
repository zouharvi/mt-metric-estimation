def get_model(args):
    if args.model == "1":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192
        model = MEModelRNN(vocab_size, 512, 128, sigmoid=True)
    elif args.model == "1l":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192
        model = MEModelRNN(vocab_size, 512, 128, sigmoid=False)
    elif args.model == "1s":
        from me_model_rnn import MEModelRNN
        vocab_size = 4096
        model = MEModelRNN(vocab_size, 256, 64, sigmoid=True)
    elif args.model == "1sv":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192
        model = MEModelRNN(vocab_size, 256, 64, sigmoid=True)
    elif args.model == "1sV":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192*2
        model = MEModelRNN(vocab_size, 256, 64, sigmoid=True)
    elif args.model == "1r":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192
        model = MEModelRNN(vocab_size, 512, 128, sigmoid=True, relu=True)
    elif args.model == "1d05":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192
        model = MEModelRNN(vocab_size, 512, 128, sigmoid=True, relu=True, dropout=0.05)
    elif args.model == "1d10":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192
        model = MEModelRNN(vocab_size, 512, 128, sigmoid=True, relu=True, dropout=0.10)
    elif args.model == "1d20":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192
        model = MEModelRNN(vocab_size, 512, 128, sigmoid=True, relu=True, dropout=0.20)
    elif args.model == "1d30":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192
        model = MEModelRNN(vocab_size, 512, 128, sigmoid=True, relu=True, dropout=0.30)
    elif args.model == "1d40":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192
        model = MEModelRNN(vocab_size, 512, 128, sigmoid=True, relu=True, dropout=0.40)
    elif args.model == "1d20l2":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192
        model = MEModelRNN(vocab_size, 512, 128, sigmoid=True, relu=True, dropout=0.20, num_layers=2)
    elif args.model == "b":
        from me_model_b import MEModelBaseline
        # not used for the model but still needs to be defined
        vocab_size = 8192
        model = MEModelBaseline()
    else:
        raise Exception("Unknown model")

    return model, vocab_size