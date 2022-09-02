#!/usr/bin/bash

# for start_n in 1 2 3 4; do
for start_n in 0; do
    end_n=$((start_n+1))
    # for langs in de-en de-pl pl-de zh-en en-zh cs-en en-cs ru-en en-ru fr-en en-fr hi-en en-hi; do
    # for direction in de-en ru-en; do
    #     echo "Submitting ${direction} ${start_n}00:${end_n}00"
    #     bsub -W 12:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py \
    #         --direction $direction \
    #         --dry-dataset \
    #         --n-start ${start_n}00 --n-end ${end_n}00 -o /dev/null --overwrite;
    # done;

    for direction in de-pl zh-en cs-en fr-en hi-en; do
        echo "Submitting ${direction} ${start_n}00:${end_n}00"
        bsub -W 12:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py \
            --direction $direction \
            --dry-dataset \
            --model helsinki \
            --n-start ${start_n}00 --n-end ${end_n}00 -o /dev/null --overwrite;
    done;

    for direction in pl-de en-zh en-cs en-fr en-hi; do
        echo "Submitting ${direction} ${start_n}00:${end_n}00"
        bsub -W 12:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py \
            --direction $direction \
            --dry-model \
            --model helsinki \
            --n-start ${start_n}00 --n-end ${end_n}00 -o /dev/null --overwrite;
    done;

    # for direction in en-ru; do
    #     echo "Submitting ${direction} ${start_n}00:${end_n}00"
    #     bsub -W 12:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py \
    #         --direction $direction \
    #         --dry-model \
    #         --n-start ${start_n}00 --n-end ${end_n}00 -o /dev/null --overwrite;
    # done;
done;