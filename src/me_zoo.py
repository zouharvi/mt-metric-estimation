def get_model(args):
    # not used by all models but still needs to be defined
    vocab_size = 8192
    if False:
        pass
    elif args.model == "1hd75b10":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192
        model = MEModelRNN(
            vocab_size, 512, 128, fusion=args.fusion,
            sigmoid=True, relu=True,
            num_layers=2, dropout=0.0, final_hidden_dropout=0.75, batch_size=10,
            load_path=args.model_load_path,
        )
    elif args.model == "1hd75b10lin":
        from me_model_rnn import MEModelRNN
        vocab_size = 8192
        model = MEModelRNN(
            vocab_size, 512, 128, fusion=args.fusion,
            sigmoid=False, relu=True,
            num_layers=2, dropout=0.0, final_hidden_dropout=0.75, batch_size=10,
            load_path=args.model_load_path,
        )
    # main model is nicknamed joist
    elif args.model in {"bdb10lin", "joist"}:
        from me_model_b_dense import MEModelBaselineDense
        vocab_size = 8192
        model = MEModelBaselineDense(sigmoid=False, batch_size=10)
    elif args.model == "joist_multi":
        from me_model_rnn_multi import MEModelRNNMulti
        vocab_size = 8192

        TARGET_METRICS = ["bleu", "bleurt", "ter", "meteor", "comet"]
        if "human" in args.data_train:
            print("Adding zscore to the optimization set")
            TARGET_METRICS.append("zscore")
        else:
            add_target_metrics_zscore = False

        model = MEModelRNNMulti(
            vocab_size, 512, 128, fusion=args.fusion,
            num_layers=2, dropout=0.0, final_hidden_dropout=0.75, batch_size=10,
            target_metrics=TARGET_METRICS,
            load_path=args.model_load_path,
        )
    elif args.model == "b":
        from me_model_b import MEModelBaseline
        model = MEModelBaseline()
    elif args.model == "comet":
        from me_model_comet import MEModelComet
        model = MEModelComet()
    else:
        raise Exception("Unknown model")

    return model, vocab_size
