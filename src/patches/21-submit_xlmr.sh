#!/usr/bin/bash

metric=zscore
echo "Submitting XLMR with $metric metric";
sbatch --time=12:00:00 --ntasks=8 --mem-per-cpu=3G --gpus=1 \
    --job-name="urbicide_1" \
    --output="lsf_logs/urbicide_1.log" \
    --wrap="\
        python3 src/run_me_model.py \
        -m xlmr --dev-n 1000 \
        -dt computed/en_de_human_metric_ft.jsonl \
        --metric ${metric} \
        -l logs/en_de_urbicide_1_${metric}.jsonl \
        --epochs 100 \
;"


metric=zscore
echo "Submitting MBERT with $metric metric";
sbatch --time=12:00:00 --ntasks=8 --mem-per-cpu=3G --gpus=1 \
    --job-name="urbicide_2" \
    --output="lsf_logs/urbicide_2.log" \
    --wrap="\
        python3 src/run_me_model.py \
        -m mbert --dev-n 1000 \
        -dt computed/en_de_human_metric_ft.jsonl \
        --metric ${metric} \
        -l logs/en_de_urbicide_2_${metric}.jsonl \
        --epochs 100 \
;"