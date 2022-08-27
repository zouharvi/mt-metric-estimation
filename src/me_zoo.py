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
    elif args.model == "1hd75":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192
        model = MEModelRNN(vocab_size, 512, 128, sigmoid=True, relu=True, dropout=0.0, final_hidden_dropout=0.75)
    elif args.model == "1d20ss12":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192
        model = MEModelRNN(vocab_size, 512, 128, sigmoid=True, relu=True, dropout=0.20, sigmoid_scale=1.2)
    elif args.model == "1hd75b10":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192
        model = MEModelRNN(
            vocab_size, 512, 128, fusion=args.fusion,
            sigmoid=True, relu=True,
            num_layers=2, dropout=0.0, final_hidden_dropout=0.75, batch_size=10
        )
    elif args.model == "1hd75b10lin":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192
        model = MEModelRNN(
            vocab_size, 512, 128, fusion=args.fusion,
            sigmoid=False, relu=True,
            num_layers=2, dropout=0.0, final_hidden_dropout=0.75, batch_size=10,
            load_path=args.model_path,
        )
    elif args.model == "bd":
        from me_model_b_dense import MEModelBaselineDense
        # not used for the model but still needs to be defined
        vocab_size = 8192
        model = MEModelBaselineDense(sigmoid=True)
    elif args.model == "bdl":
        from me_model_b_dense import MEModelBaselineDense
        # not used for the model but still needs to be defined
        vocab_size = 8192
        model = MEModelBaselineDense(sigmoid=False)
    elif args.model == "bdb10":
        from me_model_b_dense import MEModelBaselineDense
        # not used for the model but still needs to be defined
        vocab_size = 8192
        model = MEModelBaselineDense(sigmoid=True, batch_size=10)
    elif args.model == "bdlb10":
        from me_model_b_dense import MEModelBaselineDense
        # not used for the model but still needs to be defined
        vocab_size = 8192
        model = MEModelBaselineDense(sigmoid=False, batch_size=10)
    elif args.model == "b":
        from me_model_b import MEModelBaseline
        # not used for the model but still needs to be defined
        vocab_size = 8192
        model = MEModelBaseline()
    elif args.model == "comet":
        from me_model_comet import MEModelComet
        # not used for the model but still needs to be defined
        vocab_size = 8192
        model = MEModelComet()
    else:
        raise Exception("Unknown model")

    return model, vocab_size