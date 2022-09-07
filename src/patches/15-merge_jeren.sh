#!/usr/bin/bash

for direction in de-en de-pl pl-de zh-en en-zh cs-en en-cs ru-en en-ru fr-en en-fr hi-en en-hi; do
    direction2=${direction/-/_};
    echo "Merging ${direction2}";
    cat computed/${direction2}_*_metric.jsonl > computed/${direction2}_metric.jsonl;
    wc -l computed/${direction2}_metric.jsonl;
done;