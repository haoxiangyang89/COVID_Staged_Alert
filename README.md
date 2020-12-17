# Overview
This software supports the results in the paper:
https://www.medrxiv.org/content/10.1101/2020.11.26.20152520v1

The algorithm simulates a COVID-19 epidemic for Austin, TX, and determines triggers
to enact social distancing orders to avoid exceeding hospital capacity. Chosen triggers
attempt to minimize the total number of days that a city is in strict levels.

## Installation and running the code
- Download and unzip the code to a local path (e.g., .../COVID_Staged_Alert)
- Add both /COVID_Staged_Alert and /COVID_Staged_Alert/InterventionsMIP to your $PYTHONPATH
- The following packages are required:
* matplotlib
* pandas
* numpy
* scipy


## Guidelines for contributing
- Create new branches to test new features
- Create a pull request to merge with master

# Structure of the code

## threshold_policy.py
- Main module to launch the search
- Functions to execute the search and find optimized thresholds to enact lock-downs.
- Iterators for traing and testing
- Calendar generation (ad-hoc for Austin instance)

## policy_search_functions.py
- Main module to run the search
- Functions to perform chance-constrained check and select the optimal policies w/o ACS setup

## SEIYHARD_sim.py:
- Simulator engine
- Parallelizarion functions
- Calander utils class (SimCalendar)

## epi_parameters.py
- Class EpiSetup to characterize the simulation and recompute contact matrices as needed.

## intervention.py
- Class intervention defining its properties and used in the simulator.
- Helper function to create multiple interventions

## utils.py
- Timing function
- Rounding functions

## intances/__init__.py
- Module summarizing the input of the simulator.
- Creates an instance of EpiSetup

# Structure of the test scripts

## main_least_squares.py

- Least-squares fit

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

## main_downsampling.py

- Donwsampling procedure (algorithm 1)

Related input files:

instances/austin/
* tiers5_ds_Final.json
* austin_test_IHT_ds.json

## main_multitier.py

- script to obtain optimal trigger policy

Related input files:

instances/austin/
* tiers5_opt_Final.json
* transmission_Final.csv

## main_multitier_acs.py
- script to obtain optimal trigger policy and ACS setup threshold

instances/austin/
* tiers5_acs_Final.json
* transmission_Final_orange25_yellow75.csv
