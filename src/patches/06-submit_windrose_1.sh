#!/usr/bin/bash

for shuffle_seed in 1 2 3 4 5; do
    for dsize in 500 1000 2000 5000 10000 14000; do
        # for metric in bleurt chrf meteor comet; do
        for metric in bleu bleurt chrf ter meteor comet; do
            echo "Submitting $metric with $dsize data size";
            bsub -W 4:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 src/run_me_model.py \
                -f 1 --dev-n 1000 --train-n $dsize \
                -dt computed/en_de_human_metric_brt.jsonl -lb models/bpe_news_500k_h1.pkl \
                -mp models/en_de_outroop_23_${metric}_${metric}_s.pt \
                --metric zscore \
                --shuffle-train $shuffle_seed \
                -l logs/en_de_windrose_1_${metric}_${dsize}_s${shuffle_seed}.jsonl \
                --epochs 1000;
        done;

        echo "Submitting zscore with $dsize data size";
        bsub -W 4:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py \
            -f 1 --dev-n 1000 --train-n $dsize \
            -dt computed/en_de_human_metric_brt.jsonl -lb models/bpe_news_500k_h1.pkl \
            -m 1hd75b10lin \
            --metric zscore \
            --shuffle-train $shuffle_seed \
            -l logs/en_de_windrose_1_zscore_${dsize}_s${shuffle_seed}.jsonl \
            --epochs 1000;
    done;
done;