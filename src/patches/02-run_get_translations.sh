bsub -W 120:00 -n 16 -R "rusage[mem=2000,ngpus_excl_p=1]" python3 ./src/get_translations.py