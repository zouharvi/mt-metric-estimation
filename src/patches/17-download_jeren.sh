#!/usr/bin/bash

scp euler:/cluster/work/sachan/vilem/mt-metric-estimation/logs/*jeren*.jsonl logs/jeren/

for f in logs/jeren/en_de_jeren_*.jsonl; do
    echo $f ${f/en_de_jeren/jeren}
    mv "$f" "${f/en_de_jeren/jeren}";
done


# copy en-de
# scp euler:/cluster/work/sachan/vilem/mt-metric-estimation/logs/en_de_outroop_25_* logs/jeren/


# for f in logs/jeren/en_de_outroop_25_*.jsonl; do
#     echo $f ${f/en_de_outroop_25_/jeren_en_de_}
#     mv "$f" "${f/en_de_outroop_25_/jeren_en_de_}";
# done