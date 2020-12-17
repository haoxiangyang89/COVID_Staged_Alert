## running demo for COVID-19 multi-tier trigger analysis

## obtain the optimal trigger policy
python3 ../InterventionsMIP/main_multitier.py austin -n_proc=4 -f=setup_data_Final.json -train_reps=1 -test_reps=1 -f_config=austin_test_IHT_r2.json -t=tiers5_ds_Final.json -tr=transmission_Final.csv -hos=austin_real_hosp_downsampling.csv -field=ToIHT -pub=200

## perform the output procedure, plotting and generating pdf report
python3 ../InterventionsMIP/pipelinemultitier.py
