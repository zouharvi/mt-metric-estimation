#!/usr/bin/bash

for direction in de-en de-pl pl-de zh-en en-zh cs-en en-cs ru-en en-ru fr-en en-fr hi-en en-hi; do
    direction2=${direction/-/_};
    echo "Running ${direction2} for bleu";
    bsub -W 4:00 -n 8 -R "rusage[mem=4000]" python3 ./src/run_me_model.py \
        -l /dev/null \
        -f 2 -m 1hd75b10lin \
        --metric bleu --metric-dev blue \
        -dt computed/${direction2}_metric.jsonl --dev-n 10000 \
        --save-bpe models/bpe_${direction2}_clipped.pkl \
        --save-bpe-only;
done;