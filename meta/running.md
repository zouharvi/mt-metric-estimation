|date|comment|command|
|-|-|-|
|08-19-2022|first 100k wmt14 sents|`python3 ./src/get_translations.py -o computed/de_en.csv` (local)|
|08-19-2022|100k-400k wmt14 sents|`bsub -W 72:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py --n-start 100 --n-end 400 -o computed/de_en_2.csv`|