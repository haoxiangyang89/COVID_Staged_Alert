# COVID_Staged_Alert
Codes and data for "Design of COVID-19 Staged Alert Systems to Ensure Healthcare Capacity with Minimal Closures"

## Least-squares fit

Script to run:

* main_least_squares.py

Related input files:

instances/austin/
* austin_real_icu_lsq.csv
* austin_real_hosp_lsq.csv
* austin_hos_ad_lsq.csv
* transmission_Final_lsq.csv
* setup_data_Final_lsq.json

instances/houston/
* houston_real_icu_lsq.csv
* houston_real_hosp_lsq.csv
* transmission_Final_lsq.csv
* setup_data_Final_lsq.json

## Donwsampling procedure (algorithm 1)

instances/austin/
* tiers5_ds_Final.json
* austin_test_IHT_ds.json

instances/houston/
* tiers5_ds_Final.json
* houston_test_IHT_ds.json

Script to run:

* main_downsampling.py
