#!/usr/bin/bash

metric=ter

for direction in en_cs fr_en; do
    echo $direction;
    bsub -W 4:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]"  python3 ./src/eval_me_model.py \
        -d computed/${direction}_metric.jsonl \
        -od computed/jeren/${direction}_predicted.jsonl \
        -mp models/en_de_jeren_${direction}_${metric}.pt \
        -m 1hd75b10lin -f 2 \
        --data-n 10000 \
        --load-bpe models/bpe_${direction}_clipped.pkl \
        --metric ter; 
done


direction="en_de"
echo $direction;
bsub -W 4:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]"  python3 ./src/eval_me_model.py \
    -d computed/${direction}_metric.jsonl \
    -od computed/jeren/${direction}_predicted.jsonl \
    -mp models/en_de_outroop_25_ter_ter.pt \
    -m 1hd75b10lin -f 2 \
    --data-n 10000 \
    --load-bpe models/bpe_news_500k_h1.pkl \
    --metric ter;