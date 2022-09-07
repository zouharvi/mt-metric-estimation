#!/usr/bin/bash

for metric in bleu bleurt chrf meteor comet ter; do
    for direction in de-pl pl-de zh-en en-zh cs-en en-cs ru-en en-ru fr-en en-fr hi-en en-hi; do
        echo $direction;
        bsub -W 0:30 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]"  python3 ./src/eval_me_model.py \
            -d computed/${direction}_metric.jsonl \
            -od computed/jeren/${direction}_${metric}_predicted.jsonl \
            -mp models/en_de_jeren_${direction}_${metric}.pt \
            -m 1hd75b10lin -f 2 \
            --data-n 10000 \
            --load-bpe models/bpe_${direction}_clipped.pkl \
            --output-corr computed/jeren_eval/${direction}_eval.jsonl \
            --metric ${metric}; 
    done;

    direction="en_de"
    echo $direction;
    bsub -W 0:30 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]"  python3 ./src/eval_me_model.py \
        -d computed/${direction}_metric.jsonl \
        -od computed/jeren/${direction}_${metric}_predicted.jsonl \
        -mp models/en_de_outroop_25_${metric}_${metric}.pt \
        -m 1hd75b10lin -f 2 \
        --data-n 10000 \
        --load-bpe models/bpe_news_500k_h1.pkl \
        --output-corr computed/jeren_eval/${direction}_eval.jsonl \
        --metric ${metric};
done;
