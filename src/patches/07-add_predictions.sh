#!/usr/bin/bash

cp computed/en_de_metric.jsonl computed/en_de_metric_pred.jsonl
cp computed/en_de_human_metric_brt.jsonl computed/en_de_human_metric_pred.jsonl

for metric in bleu bleurt chrf ter meteor comet; do
    echo "Running $metric";
    ./src/eval_me_model.py \
        -d computed/en_de_metric_pred.jsonl \
        -od computed/en_de_metric_pred.jsonl \
        -mp models/en_de_outroop_25_${metric}_${metric}.pt \
        -dn 10000 --metric $metric;
done

for metric in bleu bleurt chrf ter meteor comet; do
    echo "Running $metric";
    # use -f 1
    ./src/eval_me_model.py \
        -d computed/en_de_human_metric_pred.jsonl \
        -f 1 \
        -od computed/en_de_human_metric_pred.jsonl \
        -mp models/en_de_outroop_26_${metric}.pt \
        -dn 1000 --metric $metric;
done

# use -f 1
./src/eval_me_model.py \
    -d computed/en_de_human_metric_pred.jsonl \
    -f 1 \
    -od computed/en_de_human_metric_pred.jsonl \
    -mp models/en_de_outroop_23_zscore_zscore_r_news.pt \
    -dn 1000 --metric zscore;