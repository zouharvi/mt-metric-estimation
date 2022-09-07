#!/usr/bin/bash

# for metric in bleu; do
# for metric in bleurt chrf meteor comet ter; do
for metric in bleu bleurt chrf meteor comet ter; do
    for direction in de-en de-pl pl-de zh-en en-zh cs-en en-cs ru-en en-ru fr-en en-fr hi-en en-hi; do
        direction2=${direction/-/_};
        echo "Running ${direction2} for ${metric}";
        bsub -W 4:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py \
            -l logs/en_de_jeren_${direction2}_${metric}.jsonl \
            -f 2 -m 1hd75b10lin \
            --metric ${metric} --metric-dev ${metric} \
            -dt computed/${direction2}_metric.jsonl --dev-n 10000;
    done;
done;