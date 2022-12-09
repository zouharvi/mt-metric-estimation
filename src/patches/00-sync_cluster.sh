#!/usr/bin/bash

rsync -azP --filter=":- .gitignore" --exclude .git/ . euler:/cluster/work/sachan/vilem/mt-metric-estimation/

# scp computed/en_de_human_train.jsonl euler:/cluster/work/sachan/vilem/mt-metric-estimation/computed/
# scp computed/en_de_metric_ft.jsonl euler:/cluster/work/sachan/vilem/mt-metric-estimation/computed/
# scp computed/en_de_human_test.jsonl euler:/cluster/work/sachan/vilem/mt-metric-estimation/computed/
# scp euler:/cluster/work/sachan/vilem/mt-metric-estimation/outputs/* outputs/