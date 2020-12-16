import pickle
import os
import pandas as pd
from datetime import datetime as dt
import numpy as np
from InterventionsMIP import load_config_file,config_path
from reporting.plotting import plot_multi_tier_sims
from reporting.report_pdf import generate_report
from reporting.output_processors import build_report
from pipelinemultitier import read_hosp, multi_tier_pipeline
import csv

def getACS_util(reg_hosp, profiles, T):
    # output the expected ACS utilization number
    useList = []
    maxUseList = []
    maxDayList = []
    capList = []
    overList = []
    noTriggered = 0
    for p in profiles:
        # time series of over regular capacity
        overCap_reg = np.maximum(np.sum(p['IHT'], axis = (1,2)) - reg_hosp, 0)
        # time series of over ACS capacity
        overCap_ACS = np.maximum(np.sum(p['IHT'], axis = (1,2)) - p["capacity"],0)
        # time series of ACS usage
        acs_usage = overCap_reg - overCap_ACS
        # time series of ACS capacity
        acs_cap = np.array(p["capacity"]) - reg_hosp
        # number of paths with ACS triggered
        if p['acs_triggered'] and len(np.unique(p['capacity'][:T])) > 1:
            noTriggered += 1
        
        # ACS required for this path
        maxUseList.append(np.max(overCap_reg[:T]))
        # number of days requiring ACS for this path
        maxDayList.append(np.sum(overCap_reg[:T] > 0))
        # total number of ACS usage for this path
        useList.append(np.sum(acs_usage[:T]))
        # total capacity of ACS usage for this path
        capList.append(np.sum(acs_cap[:T]))
        # total number of ACS unsatisfaction for this path
        overList.append(np.sum(overCap_ACS[:T]))
    meanUse = np.mean(useList)
    meanUtil = np.nanmean(np.array(useList)/np.array(capList))
    return useList,maxUseList,maxDayList,capList,overList,meanUse,meanUtil,noTriggered

def getACS_reppath(profiles,reg_cap):
    dateList = []
    for i in range(300):
        if len(np.where(np.array(profiles[i]['capacity']) > reg_cap)[0]) > 0:
            dateList.append([i,np.where(np.array(profiles[i]['capacity']) > reg_cap)[0][0]])
        else:
            dateList.append([i,10000])
    dateList.sort(key = lambda x: x[1]) 
    return dateList
    
def getACS_gap(profiles, reg_cap):
    outList = []
    for i in range(300):
        IHTList = np.sum(profiles[i]['IHT'], axis = (1,2))
        capList = np.array(profiles[i]['capacity'])
        profList = [i]
        if len(np.where(IHTList > reg_cap)[0]) > 0:
            profList.append(np.where(IHTList > reg_cap)[0][0])
        else:
            profList.append(10000)
        if len(np.where(capList > reg_cap)[0]) > 0:
            profList.append(np.where(capList > reg_cap)[0][0])
        else:
            profList.append(10000)
        outList.append(profList)
    return outList

#%%

fileList = os.listdir("output/ACS_Analysis")

# load Austin real hospitalization
file_path = "instances/austin/austin_real_hosp_updated.csv"
start_date = dt(2020,2,28)
real_hosp = read_hosp(file_path, start_date)
hosp_beds_list = None
file_path = "instances/austin/austin_hosp_ad_updated.csv"
hosp_ad = read_hosp(file_path, start_date, "admits")
file_path = "instances/austin/austin_real_icu_updated.csv"
real_icu = read_hosp(file_path, start_date)
hosp_beds_list = None

# newly defined color
add_tiers = {0.62379925: '#ffE000',
             0.6465315: '#ffC000',
             0.66926375: '#ffA000',
             0.71472825: '#ff6000',
             0.7374605: '#ff4000',
             0.76019275: '#ff2000'
    }

fi = open("/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/output/ACS_Analysis.csv","w",newline="")
csvWriter = csv.writer(fi,dialect='excel')
csvWriter.writerow(['Case_Name','ACS_Quantity','ACS_Trigger','Scenario with ACS Triggered',
                    'Infeasible Scenarios','Mean ACS Usage', 'Mean ACS Util Rate', 
                    'Max No of Days Requiring ACS','95% Days Requiring ACS',
                    'Max ACS Required', '95% ACS Required', 'Original Unmet Mean', 'Original Unmet Median', 'Original Unmet Std', 'Original Unmet 5%', 'Original Unmet 95%'])

trend_comp = True

for instance_raw in fileList:
    if ".p" in instance_raw:
        try:
            instance_name = instance_raw[:-2]
            file_path = f'output/ACS_Analysis/{instance_name}.p'
            with open(file_path, 'rb') as outfile:
                read_output = pickle.load(outfile)
            instance, interventions, best_params, best_policy, best_sim, profiles, config, cost_record, seeds_info = read_output
            
            acs_results = getACS_util(instance.hosp_beds,profiles,580)
            print("====================================================")
            print(instance_name)
            case_name = str(instance.transmission_file)[68:-4]
            print(case_name)
            instance_name = "austin_{}_{}".format(case_name,best_policy.acs_Q)
            os.rename(file_path, r"/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/output/ACS_Analysis/" + instance_name + ".p")
            
            if "11_1" in instance_name:
                end_history = dt(2020,11,1)
            else:
                end_history = dt(2020,10,7)
            t_start = (end_history - start_date).days

            print("ACS Trigger: ", best_policy.acs_thrs)
            print("ACS Quantity: ", best_policy.acs_Q)
            
            infeas_scen = np.sum(np.array(acs_results[4]) > 0)
            print("Infeasible Scenarios Testing: ", infeas_scen)
            print("Mean ACS Usage: ", acs_results[5])
            
            mean_util_rate = np.round(acs_results[6]*100,2)
            print("Mean ACS Utilization Rate: ", mean_util_rate)
            print("Number of paths hitting the trigger: ", acs_results[7])
            
            print("Maximum number of days requiring ACS", np.max(acs_results[2]))
            print("95 Percentile of days requiring ACS", np.percentile(acs_results[2],95))
            print("Maximum ACS required", np.max(acs_results[1]))
            print("95 Percentile of ACS required", np.percentile(acs_results[1],95))
            
            n_replicas = len(profiles)
            unmet_IHT = [np.sum(np.maximum(np.sum(profiles[i]['IHT'],axis = (1,2)) - 1500,0)) for i in range(300)]
            over_mean = np.mean(unmet_IHT)
            over_median = np.median(unmet_IHT)
            over_std = np.std(unmet_IHT)
            over_5P = np.percentile(unmet_IHT,5)
            over_95P = np.percentile(unmet_IHT,95)
            
            data = [case_name, best_policy.acs_Q, best_policy.acs_thrs, acs_results[7],
                    infeas_scen, acs_results[5], mean_util_rate, 
                    np.max(acs_results[2]), np.percentile(acs_results[2],95),
                    np.max(acs_results[1]), np.percentile(acs_results[1],95),over_mean,over_median,over_std,over_5P,over_95P]
            csvWriter.writerow(data)
            dateList = getACS_reppath(profiles,1500)
            if not trend_comp:
                cpid = dateList[14][0]
            else:
                cpid = 0
            IHD_plot = plot_multi_tier_sims(instance_name,
                            instance,
                            best_policy,
                            profiles, ['sim'] * len(profiles),
                            real_hosp,
                            plot_left_axis=['IHT'],
                            plot_right_axis=[],
                            T=580,
                            interventions=interventions,
                            show=False,
                            align_axes=True,
                            plot_triggers=False,
                            plot_trigger_annotations=False,
                            plot_legend=False,
                            y_lim=best_policy.acs_Q + 2000,
                            policy_params=best_params,
                            n_replicas=n_replicas,
                            config=config,
                            add_tiers=add_tiers,
                            t_start = t_start,
                            central_path_id = cpid,
                            cap_path_id = cpid,
                            history_white = True,
                            acs_fill = True
                            )
            IYIH_plot = plot_multi_tier_sims(instance_name,
                                  instance,
                                  best_policy,
                                  profiles, ['sim'] * len(profiles),
                                  real_hosp,
                                  plot_left_axis=['ToIHT'],
                                  plot_right_axis=[],
                                  T=580,
                                  interventions=interventions,
                                  show=False,
                                  align_axes=False,
                                  plot_triggers=False,
                                  plot_ACS_triggers=True,
                                  plot_trigger_annotations=False,
                                  plot_legend=False,
                                  y_lim=None,
                                  policy_params=best_params,
                                  n_replicas=n_replicas,
                                  config=config,
                                  hosp_beds_list=hosp_beds_list,
                                  real_new_admission=hosp_ad,
                                  add_tiers=add_tiers,
                                  t_start = t_start,
                                  central_path_id = cpid,
                                  cap_path_id = cpid,
                                  history_white = True
                                  )
        except:
            pass
        
fi.close()