#!/usr/bin/bash

# for start_n in 0 1 2 3 4; do
for start_n in 1 2 3 4; do
    end_n=$((start_n+1))
    for model in w16g w16t w17c t5; do
        echo "Submitting ${model} ${start_n}00:${end_n}00"
        bsub -W 48:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py \
            --direction en-de \
            -m $model \
            --n-start ${start_n}00 --n-end ${end_n}00 -o computed/en_de_${model}_${start_n}.jsonl
    done;
done;