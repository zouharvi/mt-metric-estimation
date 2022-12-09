#!/usr/bin/bash

# MBERT direct
metric=zscore
echo "Submitting MBERT with $metric metric";
sbatch --time=12:00:00 --ntasks=8 --mem-per-cpu=3G --gpus=1 \
    --job-name="censer_1 (direct)" \
    --output="lsf_logs/censer_1.log" \
    --wrap="\
        python3 src/run_me_model.py \
        -m mbert --dev-n 1000 \
        -dt computed/en_de_human_metric_ft.jsonl \
        --metric ${metric} \
        -l logs/en_de_censer_1_${metric}.jsonl \
        --epochs 100 \
;"

# pre train
metric=ter
echo "Submitting MBERT with $metric metric";
sbatch --time=4-0 --ntasks=8 --mem-per-cpu=3G --gpus=1 \
    --job-name="censer_2 (pretrain)" \
    --output="lsf_logs/censer_2.log" \
    --wrap="\
        python3 src/run_me_model.py \ 
        -m mbert --dev-n 1000 \
        -dt computed/en_de_metric_ft.jsonl \
        --metric ${metric} \
        -l logs/en_de_censer_2_${metric}.jsonl \
        --epochs 100 \
;"


# pre train
metric=ter
echo "Submitting JOIST with $metric metric";
sbatch --time=4-0 --ntasks=8 --mem-per-cpu=3G --gpus=1 \
    --job-name="censer_5 (pretrain)" \
    --output="lsf_logs/censer_5.log" \
    --wrap="\
        python3 src/run_me_model.py \
        -m joist --dev-n 1000 \
        -f 0 \
        -dt computed/en_de_metric_ft.jsonl \
        --metric ${metric} \
        -l logs/en_de_censer_5_${metric}.jsonl \
        --epochs 100 \
;"


# fine tune
metric=zscore
echo "Submitting MBERT with $metric metric";
sbatch --time=4-0 --ntasks=8 --mem-per-cpu=3G --gpus=1 \
    --job-name="censer_3 (finetune)" \
    --output="lsf_logs/censer_3.log" \
    --wrap="\
        python3 src/run_me_model.py \
        -m mbert --dev-n 1000 \
        -dt computed/en_de_human_metric_ft.jsonl \
        --metric ${metric} \
        -l logs/en_de_censer_3_${metric}.jsonl \
        --epochs 100 \
;"

# infer
metric=zscore
echo "Submitting infer";
sbatch --time=12:00:00 --ntasks=8 --mem-per-cpu=3G --gpus=1 \
    --job-name="censer_infer" \
    --output="lsf_logs/censer_infer.log" \
    --wrap="\
        python3 src/infer_me_model.py \
        -m mbert  \
        -mp models/en_de_censer_1_${metric}.pt \
        -d computed/en_de_human_test.jsonl \
        -o output/en_de_human_censer_1.csv \
        -l logs/en_de_censer_infer_${metric}_finetuned.jsonl \
;"