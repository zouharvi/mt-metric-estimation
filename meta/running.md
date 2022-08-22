|date|status|nickname|comment|command|
|-|-|-|-|-|
|08-20-2022|running|outroop_12|fusion multi, relu, d 20%, 500k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_15.jsonl -f 1 -m 1d20` (local)|
|08-20-2022|ok|outroop_14|fusion multi, relu, d 30%, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_14.jsonl -f 1 -m 1d40` (local)|
|08-20-2022|ok|outroop_13|fusion multi, relu, d 40%, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_13.jsonl -f 1 -m 1d30` (local)|
|08-20-2022|ok|outroop_12|fusion multi, relu, d 20%, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_12.jsonl -f 1 -m 1d20` (local)|
|08-20-2022|ok|outroop_11|fusion multi, relu, d 10%, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_11.jsonl -f 1 -m 1d10` (local)|
|08-20-2022|ok|outroop_10|fusion multi, relu, d 5%, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_10.jsonl -f 1 -m 1d05` (local)|
|08-20-2022|ok|outroop_9|fusion multi, relu, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_9.jsonl -f 1 -m 1sV` (local)|
|08-20-2022|ok|outroop_8|fusion multi, relu, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_8.jsonl -f 1 -m 1r` (local)|
|08-20-2022|ok|outroop_7|fusion multi, small, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_7.jsonl -f 1 -m 1sv` (local)|
|08-20-2022|ok|outroop_6|fusion multi, small, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_6.jsonl -f 1 -m 1s` (local)|
|08-20-2022|ok|outroop_5|fusion multi, linear, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_5.jsonl -f 1 -m 1l` (local)|
|08-20-2022|ok|outroop_4|fusion multi, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_4.jsonl -f 1` (local)|
|08-20-2022|ok|somnorif_0|baseline, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_somnorif_0.jsonl --model b` (local)|
|08-20-2022|ok||400k-500k wmt14 sents|`bsub -W 48:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py --direction en-de --n-start 400 --n-end 500 -o logs/en_de_4.csv`|
|08-20-2022|ok||300k-400k wmt14 sents|`bsub -W 48:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py --direction en-de --n-start 300 --n-end 400 -o logs/en_de_3.csv`|
|08-20-2022|ok||200k-300k wmt14 sents|`bsub -W 48:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py --direction en-de --n-start 200 --n-end 300 -o logs/en_de_2.csv`|
|08-20-2022|ok||100k-200k wmt14 sents|`bsub -W 48:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py --direction en-de --n-start 100 --n-end 200 -o logs/en_de_1.csv`|
|08-20-2022|ok||000k-100k wmt14 sents|`bsub -W 48:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py --direction en-de --n-start 0 --n-end 100 -o logs/en_de_0.csv`|
|08-20-2022|ok|outroop_3|fusion conf + exp(conf), 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_3.jsonl -f 1` (local)|
|08-19-2022|ok|outroop_2|no fusion, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_2.jsonl` (local)|
|08-20-2022|ok||400k-500k wmt14 sents|`bsub -W 48:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py --n-start 400 --n-end 500 -o logs/de_en_5.csv`|
|08-20-2022|ok||300k-400k wmt14 sents|`bsub -W 48:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py --n-start 300 --n-end 400 -o logs/de_en_4.csv`|
|08-20-2022|ok||200k-300k wmt14 sents|`bsub -W 48:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py --n-start 200 --n-end 300 -o logs/de_en_3.csv`|
|08-19-2022|ok (stopped)||100k-400k wmt14 sents|`bsub -W 72:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py --n-start 100 --n-end 400 -o logs/de_en_2.csv`|
|08-19-2022|ok||first 100k wmt14 sents|`python3 ./src/get_translations.py -o logs/de_en.csv` (local)|
|08-19-2022|ok|outroop_1|no fusion, 40k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_1.jsonl` (local)|