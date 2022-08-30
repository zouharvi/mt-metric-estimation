#!/usr/bin/bash

for metric in bleu bleurt chrf ter meteor comet; do
    echo "Submitting mbert with $metric metric";
    bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=12000]" python3 src/run_me_model.py \
        -m mbert --dev-n 10000 \
        -dt computed/en_de_metric_ft.jsonl \
        --metric ${metric} \
        -l logs/en_de_zepole_1_${metric}.jsonl \
        --epochs 100;
done;

metric=zscore
echo "Submitting mbert with $metric metric";
bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=12000]" python3 src/run_me_model.py \
    -m mbert --dev-n 1000 \
    -dt computed/en_de_human_metric_ft.jsonl \
    --metric ${metric} \
    -l logs/en_de_zepole_1_${metric}.jsonl \
    --epochs 100;