#!/usr/bin/bash

for model in w16g w16t w17c t5; do
    echo "Submitting ${model} -f 2"
    bsub -W 24:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py \
        -l logs/en_de_trema_0_${model}_bleu.jsonl \
        -f 2 -m 1hd75b10lin \
        --metric bleu \
        -dt computed/en_de_${model}_metric.jsonl \
        --dev-n 10000;

    echo "Submitting ${model} -f 0"
    bsub -W 24:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py \
        -l logs/en_de_trema_1_${model}_bleu.jsonl \
        -f 0 -m 1hd75b10lin \
        --metric bleu \
        -dt computed/en_de_${model}_metric.jsonl \
        --dev-n 10000;
done;