
|date|status|nickname|comment|command|
|-|-|-|-|-|
|08-28-2022|ok|outroop_26|(fusion 1, human data, metrics)|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -dt computed/en_de_human_metric_brt.jsonl --dev-n 1000 -f 1 -m 1hd75b10lin -l logs/en_de_outroop_26_bleu.jsonl --metric bleu`|
|08-28-2022|running|racoon_1|multi (autometrics)|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 src/run_me_model.py -m joist_multi -dt computed/en_de_metric_brt.jsonl --dev-n 10000 -lb models/bpe_news_500k_h1.pkl  -l logs/en_de_racoon_1.jsonl`|
|08-28-2022|running|racoon_0|multi (all)|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 src/run_me_model.py -m joist_multi -dt computed/en_de_human_metric_brt.jsonl --dev-n 1000 -lb models/bpe_news_500k_h1.pkl  -l logs/en_de_racoon_0.jsonl`|
|08-28-2022|running|outroop_25|x->x (fusion 2, metrics)|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/en_de_outroop_25_bleu_bleu.jsonl -f 2 -m 1hd75b10lin --metric bleu --metric-dev bleu -dt computed/en_de_metric.jsonl --dev-n 10000`|
|08-28-2022|ok|outroop_{23,24}_news|zscore|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/en_de_outroop_23_zscore_zscore_r_news.jsonl -f 1 -m 1hd75b10lin --metric bleu --metric-dev bleu -dt computed/en_de_human_metric.jsonl --dev-n 1000 --epochs 110  -lb models/bpe_news_500k_h1.pkl`|
|08-27-2022|ok|windrose_0|finetuning (all metrics)|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" ./src/run_me_model.py -f 1 --dev-n 1000 -dt computed/en_de_human_metric_fixed.jsonl -lb models/bpe_news_500k_h1.pkl -mp models/en_de_outroop_23_bleu_bleu_s.pt --metric zscore --metric-dev zscore -l logs/en_de_windrose_0_bleu.jsonl --epochs 1000`|
|08-27-2022|running||metrics h4|`bsub -W 120:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_metric.py -i computed/en_de_h4.jsonl -o computed/en_de_h4_metric.jsonl`|
|08-27-2022|running||metrics h3|`bsub -W 120:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_metric.py -i computed/en_de_h3.jsonl -o computed/en_de_h3_metric.jsonl`|
|08-27-2022|running||metrics h5 (longer)|`bsub -W 120:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_metric.py -i computed/en_de_h5.jsonl -o computed/en_de_h5l_metric.jsonl`|
|08-27-2022|ok (removed)|outroop_23_r|zscore|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/en_de_outroop_23_zscore_zscore_r.jsonl -f 1 -m 1hd75b10lin --metric bleu --metric-dev bleu -dt computed/en_de_human_metric.jsonl --dev-n 1000 --epochs 110`|
|08-27-2022|ok|outroop_23_s|x->x (save, metrics)|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/en_de_outroop_23_bleu_bleu_s.jsonl -f 1 -m 1hd75b10lin --metric bleu --metric-dev bleu -dt computed/en_de_metric.jsonl --dev-n 10000`|
|08-27-2022|ok|somnorif_4|baseline (rerun human)|`python3 ./src/run_me_model.py -l logs/en_de_somnorif_4_METRIC.jsonl --model b --metric METRIC --dev-n 1000` (local)|
|08-27-2022|ok|hopsack_0|qe data (rerun, to merge)|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/en_de_hopsack_0.jsonl -m comet -dt computed/en_de_human_metric.jsonl --dev-n 1000`|
|08-27-2022|ok|outroop_23_r|x->x (metrics+zscore, submitted with wrong logfile direction, better than 20?)|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/en_de_outroop_23_bleu_bleu_r.jsonl -f 1 -m 1hd75b10lin --metric bleu --metric-dev bleu -dt computed/en_de_metric.jsonl --dev-n 10000`|
|08-27-2022|ok|dwile_3|bleu limited data h5 (todo 500k)|`bsub -W 72:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/en_de_dwile_3_1k.jsonl -f 1 -m 1hd75b10lin --metric bleu --metric-dev bleu -dt computed/en_de_h5_metric.jsonl --dev-n 10000 --train-n 1000 -hn 5`|
|08-27-2022|ok|dwile_2|bleu limited data h2|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/en_de_dwile_2_1k.jsonl -f 1 -m 1hd75b10lin --metric bleu --metric-dev bleu -dt computed/en_de_h2_metric.jsonl --dev-n 10000 --train-n 1000 -hn 2`|
|08-26-2022|ok||metrics h2|`bsub -W 24:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_metric.py -i computed/en_de_h2.jsonl -o computed/en_de_h2_metric.jsonl`|
|08-26-2022|killed||metrics h5|`bsub -W 24:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_metric.py -i computed/en_de_h5.jsonl -o computed/en_de_h5_metric.jsonl`|
|08-26-2022|ok|dwile_1|bleu limited data|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/en_de_dwile_0_1k.jsonl -m b --metric bleu --metric-dev bleu -dt computed/en_de_metric.jsonl --dev-n 10000 --train-n 1000`|
|08-26-2022|ok|dwile_0|bleu limited data|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/en_de_dwile_0_1k.jsonl -f 1 -m 1hd75b10lin --metric bleu --metric-dev bleu -dt computed/en_de_metric.jsonl --dev-n 10000 --train-n 1000`|
|08-26-2022|running|outroop_24|x -> x (zscore 1k, others 10k)|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/en_de_outroop_23_bleu_bleu.jsonl -f 0 -m 1hd75b10lin --metric bleu --metric-dev bleu -dt computed/en_de_metric.jsonl --dev-n 10000`|
|08-26-2022|ok|hopsack_1 me data||`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/en_de_hopsack_1.jsonl -m comet -dt computed/en_de_metric.jsonl --dev-n 10000`|
|08-26-2022|ok (accidentially on 1k human?)|outroop_23|x -> x (non zscore)|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/en_de_outroop_23_bleu_bleu.jsonl -f 1 -m 1hd75b10lin --metric bleu --metric-dev bleu -dt computed/en_de_metric.jsonl --dev-n 10000`|
|08-26-2022|ok|outroop_23|bleu,comet,zscore -> zscore|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/en_de_outroop_23_comet_zscore.jsonl -f 1 -m 1hd75b10lin --metric comet --metric-dev zscore -dt computed/en_de_human_metric.jsonl --dev-n 1000`|
|08-25-2022|ok|hopsack_0||`python3 ./src/run_me_model.py -l logs/en_de_hopsack_hopsack_0.jsonl -m comet --metric METRIC`|
|08-25-2022|ok||metrics part|`bsub -W 12:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_metric.py -i computed/en_de_4.jsonl -o computed/en_de_4_metric.jsonl`|
|08-25-2022|ok||metrics part|`bsub -W 12:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_metric.py -i computed/en_de_3.jsonl -o computed/en_de_3_metric.jsonl`|
|08-25-2022|ok||metrics part|`bsub -W 12:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_metric.py -i computed/en_de_2.jsonl -o computed/en_de_2_metric.jsonl`|
|08-25-2022|ok||metrics part|`bsub -W 12:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_metric.py -i computed/en_de_1.jsonl -o computed/en_de_1_metric.jsonl`|
|08-25-2022|ok||metrics part|`bsub -W 12:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_metric.py -i computed/en_de_0.jsonl -o computed/en_de_0_metric.jsonl`|
|08-23-2022|ok|somnorif_4|baseline|`python3 ./src/run_me_model.py -l logs/en_de_somnorif_4_METRIC.jsonl --model b --metric METRIC` (local)|
|08-23-2022|ok||400k-500k wmt14 sents|`bsub -W 48:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py --direction en-de --n-start 400 --n-end 500 -o computed/en_de_4.jsonl`|
|08-23-2022|ok||300k-400k wmt14 sents|`bsub -W 48:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py --direction en-de --n-start 300 --n-end 400 -o computed/en_de_3.jsonl`|
|08-23-2022|ok||200k-300k wmt14 sents|`bsub -W 48:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py --direction en-de --n-start 200 --n-end 300 -o computed/en_de_2.jsonl`|
|08-23-2022|ok||100k-200k wmt14 sents|`bsub -W 48:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py --direction en-de --n-start 100 --n-end 200 -o computed/en_de_1.jsonl`|
|08-23-2022|ok||000k-100k wmt14 sents|`bsub -W 48:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_translations.py --direction en-de --n-start 0 --n-end 100 -o computed/en_de_0.jsonl` (remote & local)|
|08-23-2022|ok|outroop_22|final hidden state dropout, 10 batch, linear, no fusion, 500k sents|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/de_en_outroop_22_METRIC.jsonl -f 0 -m 1hd75b10lin --metric METRIC`|
|08-23-2022|ok|outroop_21|final hidden state dropout, 10 batch, no fusion, 500k sents|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/de_en_outroop_21_METRIC.jsonl -f 0 -m 1hd75b10 --metric METRIC`|
|08-23-2022|ok|outroop_20|final hidden state dropout, 10 batch, linear, 500k sents|`bsub -W 12:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/de_en_outroop_20_METRIC.jsonl -f 1 -m 1hd75b10lin --metric METRIC`|
|08-23-2022|ok|somnorif_3|baseline, 500k sents|`python3 ./src/run_me_model.py -l logs/de_en_somnorif_3_METRIC.jsonl --model b --metric METRIC` (local)|
|08-23-2022|ok|outroop_19|final hidden state dropout, 10 batch, 500k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_19_METRIC.jsonl -f 1 -m 1hd75b10 --metric METRIC` (local)|
|08-22-2022|ok|somnorif_2c|baseline, 500k sents|`python3 ./src/run_me_model.py -l logs/de_en_somnorif_2.jsonl --model b --metric chrf` (local)|
|08-22-2022|ok||metrics all, 500k sents|`bsub -W 48:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/get_metric.py`|
|08-22-2022|ok|nephelo_3|baseline, 500k sents|`python3 ./src/run_me_model.py -l logs/de_en_nephelo_3.jsonl --model bdlb10` (local)|
|08-22-2022|ok|nephelo_2|baseline, 500k sents|`python3 ./src/run_me_model.py -l logs/de_en_nephelo_2.jsonl --model bdb10` (local)|
|08-22-2022|ok|nephelo_1|baseline, 500k sents|`python3 ./src/run_me_model.py -l logs/de_en_nephelo_1.jsonl --model bdl` (local)|
|08-22-2022|ok|nephelo_0|baseline, 500k sents|`python3 ./src/run_me_model.py -l logs/de_en_nephelo_0.jsonl --model bd` (local)|
|08-22-2022|ok|outroop_18 (bad fusion)|d 20%, sigmoid multiply, 500k sents|`bsub -W 12:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/de_en_outroop_18.jsonl -f 1 -m 1d20ss12`|
|08-22-2022|ok|outroop_17 (bad fusion)|final hidden state dropout, 500k sents|`bsub -W 12:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/de_en_outroop_17.jsonl -f 1 -m 1hd75`|
|08-22-2022|ok|somnorif_1|baseline, 500k sents|`python3 ./src/run_me_model.py -l logs/de_en_somnorif_1.jsonl --model b` (local)|
|08-22-2022|ok|outroop_16 (bad fusion)|fusion multi, relu, d 20%, 2 layers, 500k sents|`bsub -W 12:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" python3 ./src/run_me_model.py -l logs/de_en_outroop_16.jsonl -f 1 -m 1d20l2`|
|08-22-2022|ok|outroop_15 (bad fusion)|fusion multi, relu, d 20%, 500k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_15.jsonl -f 1 -m 1d20` (local)|
|08-21-2022|ok|outroop_14 (bad fusion)|fusion multi, relu, d 30%, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_14.jsonl -f 1 -m 1d40` (local)|
|08-21-2022|ok|outroop_13 (bad fusion)|fusion multi, relu, d 40%, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_13.jsonl -f 1 -m 1d30` (local)|
|08-21-2022|ok|outroop_12 (bad fusion)|fusion multi, relu, d 20%, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_12.jsonl -f 1 -m 1d20` (local)|
|08-21-2022|ok|outroop_11 (bad fusion)|fusion multi, relu, d 10%, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_11.jsonl -f 1 -m 1d10` (local)|
|08-21-2022|ok|outroop_10 (bad fusion)|fusion multi, relu, d 5%, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_10.jsonl -f 1 -m 1d05` (local)|
|08-20-2022|ok|outroop_9 (bad fusion?)|fusion multi, relu, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_9.jsonl -f 1 -m 1sV` (local)|
|08-20-2022|ok|outroop_8 (bad fusion?)|fusion multi, relu, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_8.jsonl -f 1 -m 1r` (local)|
|08-20-2022|ok|outroop_7 (bad fusion?)|fusion multi, small, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_7.jsonl -f 1 -m 1sv` (local)|
|08-20-2022|ok|outroop_6 (bad fusion?)|fusion multi, small, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_6.jsonl -f 1 -m 1s` (local)|
|08-20-2022|ok|outroop_5 (bad fusion?)|fusion multi, linear, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_5.jsonl -f 1 -m 1l` (local)|
|08-20-2022|ok|outroop_4 (bad fusion?)|fusion multi, 150k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_4.jsonl -f 1` (local)|
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
|08-19-2022|ok|outroop_1|no fusion, 40k sents|`python3 ./src/run_me_model.py -l logs/de_en_outroop_1.jsonl` (local) 