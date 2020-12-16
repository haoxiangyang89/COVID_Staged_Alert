import pickle
import os
import pandas as pd
from datetime import datetime as dt
import numpy as np
from InterventionsMIP import load_config_file,config_path
from reporting.plotting import plot_multi_tier_sims
from reporting.report_pdf import generate_report
from reporting.output_processors import build_report
from pipelinemultitier import read_hosp, multi_tier_pipeline, icu_pipeline
import csv

#%%
# script to process ICU data
#fileList = os.listdir("output/ICU")
fileList = os.listdir("output/ICU_Fixed")

# load Austin real hospitalization
file_path = "instances/austin/austin_real_icu_7_6.csv"
start_date = dt(2020,2,28)
real_hosp = read_hosp(file_path, start_date)
hosp_beds_list = None

perc_results = np.zeros((3,13,8))

ninetyfiveq_results = np.zeros((3,13,8))

capList = [1100,1050,945,830]
#capList = [1580,1500,1350,1185]
ratioList = [0.3009090909090909,0.31523809523809526,0.3502645502645503,0.39879518072289155]
#ratioList = [0.30063291139240506, 0.31666666666666665, 0.35185185185185186,0.4008438818565401]

transmission_file_list = ["transmission_625.csv",
                          "transmission_Fixed_ICU_orange.csv",
                          "transmission_Fixed_ICU_orange_red_1.csv",
                          "transmission_Fixed_ICU_orange_red_2.csv",
                          "transmission_Fixed_ICU_orange_red_3.csv",
                          "transmission_Fixed_ICU_orange_red_4.csv",
                          "transmission_Fixed_ICU_orange_red_1_lim.csv",
                          "transmission_Fixed_ICU_orange_red_2_lim.csv",
                          "transmission_Fixed_ICU_orange_red_3_lim.csv",
                          "transmission_Fixed_ICU_orange_red_4_lim.csv",
                          "transmission_76.csv",
                          "transmission_Fixed_ICU_orange_622.csv",
                          "transmission_Fixed_622_Opt.csv"
    ]

for instance_raw in fileList:
    if ".p" in instance_raw:
        try:
            instance_name = instance_raw[:-2]
            file_path = f'output/ICU/{instance_name}.p'
            with open(file_path, 'rb') as outfile:
                read_output = pickle.load(outfile)
            instance, interventions, best_params, best_policy, best_sim, profiles, config, cost_record, seeds_info = read_output
            
            # Get only desired profiles
            if real_hosp is None:
                real_hosp = instance.cal.real_hosp
            last_day_hosp_data = len(real_hosp) - 1
            lb_hosp = real_hosp[-1] * (1 - config['div_filter_frac'])
            ub_hosp = real_hosp[-1] * (1 + config['div_filter_frac'])
            profiles = [p for p in profiles]
            # if config['det_history']:
            #     profiles = [p for p in profiles]
            # else:
            #     profiles = [p for p in profiles if lb_hosp < np.sum(p['IH'][last_day_hosp_data]) < ub_hosp or p['seed'] == -1]
            n_replicas = len(profiles)
            T = np.minimum(instance.T, instance.T)  #229
            
            if np.size(instance.epi._gamma_IH) == 25:
                # new specific
                file_type = "new_Specific"
                type_ind = 2
                columnNo = capList.index(instance.hosp_beds)
                rowNo = transmission_file_list.index(str(instance.transmission_file).split("/")[-1])
            elif instance.epi._gamma_IH.det_val == 7.9:
                # new ind
                file_type = "new_Ind"
                type_ind = 1
                columnNo = capList.index(instance.hosp_beds)
                rowNo = transmission_file_list.index(str(instance.transmission_file).split("/")[-1])
            else:
                # old params
                file_type = "old"
                type_ind = 0
                if instance.hosp_beds == 1105:
                    instance.hosp_beds = 1100
                columnNo = capList.index(instance.hosp_beds)
                rowNo = transmission_file_list.index(str(instance.transmission_file).split("/")[-1])
        
            # obtain the statistics
            ICUList = []
            for p in profiles:
                ICUList.append(ratioList[columnNo]*np.max(np.sum(p['IH'],axis = (1,2))))
            perc_results[type_ind,rowNo,columnNo] = np.sum(np.array(ICUList) > 331)/300
            ninetyfiveq_results[type_ind,rowNo,columnNo] = np.quantile(ICUList,0.95)
            
            # rename the original file
            instance_name = "austin_{}_{}_{}".format(file_type,instance.hosp_beds,rowNo)
            os.rename(file_path, r"/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/output/ICU/" + instance_name + ".p")
            
            # draw the figure
            IHD_plot = plot_multi_tier_sims(instance_name,
                                        instance,
                                        best_policy,
                                        profiles, ['sim'] * len(profiles),
                                        real_hosp,
                                        plot_left_axis=['IH'],
                                        plot_right_axis=[],
                                        T=T,
                                        interventions=interventions,
                                        show=True,
                                        align_axes=True,
                                        plot_triggers=False,
                                        plot_trigger_annotations=False,
                                        plot_legend=False,
                                        y_lim=500,
                                        policy_params=best_params,
                                        n_replicas=n_replicas,
                                        config=config,
                                        hosp_beds_list=[331],
                                        bed_scale=ratioList[columnNo]
                                        )
        except:
            pass
        # IYIH_plot = plot_multi_tier_sims(instance_name,
        #                                   instance,
        #                                   best_policy,
        #                                   profiles, ['sim'] * len(profiles),
        #                                   real_hosp,
        #                                   plot_left_axis=['IYIH'],
        #                                   plot_right_axis=[],
        #                                   T=T,
        #                                   interventions=interventions,
        #                                   show=True,
        #                                   align_axes=False,
        #                                   plot_triggers=True,
        #                                   plot_trigger_annotations=False,
        #                                   plot_legend=False,
        #                                   y_lim=None,
        #                                   policy_params=best_params,
        #                                   n_replicas=n_replicas,
        #                                   config=config,
        #                                   hosp_beds_list=hosp_beds_list,
        #                                   real_new_admission=hosp_ad)


#%%
            
fileList = os.listdir("/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/output/ICU_Fixed")

# load Austin real hospitalization
file_path = "/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/instances/austin/austin_real_icu_7_6.csv"
start_date = dt(2020,2,28)
real_hosp = read_hosp(file_path, start_date)
hosp_beds_list = None

perc_results_331 = np.zeros((3,3,8))
perc_results_475 = np.zeros((3,3,8))

ninetyfiveq_results = np.zeros((3,3,8))

capList = [1100,1050,945,830,1580,1500,1350,1185]
ratioList = [0.3009090909090909,0.31523809523809526,0.3502645502645503,0.39879518072289155,0.30063291139240506, 0.31666666666666665, 0.35185185185185186,0.4008438818565401]

transmission_file_list = ["transmission_625.csv",
                          # "transmission_Fixed_ICU_orange.csv",
                          # "transmission_Fixed_ICU_orange_red_1.csv",
                          # "transmission_Fixed_ICU_orange_red_2.csv",
                          # "transmission_Fixed_ICU_orange_red_3.csv",
                          # "transmission_Fixed_ICU_orange_red_4.csv",
                          # "transmission_Fixed_ICU_orange_red_1_lim.csv",
                          # "transmission_Fixed_ICU_orange_red_2_lim.csv",
                          # "transmission_Fixed_ICU_orange_red_3_lim.csv",
                          # "transmission_Fixed_ICU_orange_red_4_lim.csv",
                          "transmission_76.csv",
                          # "transmission_Fixed_ICU_orange_622.csv",
                          "transmission_Fixed_622_Opt.csv"
    ]

for instance_raw in fileList:
    if ".p" in instance_raw:
        try:
            instance_name = instance_raw[:-2]
            file_path = f'/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/output/ICU_Fixed/{instance_name}.p'
            with open(file_path, 'rb') as outfile:
                read_output = pickle.load(outfile)
            instance, interventions, best_params, best_policy, best_sim, profiles, config, cost_record, seeds_info = read_output
            
            # Get only desired profiles
            if real_hosp is None:
                real_hosp = instance.cal.real_hosp
            last_day_hosp_data = len(real_hosp) - 1
            lb_hosp = real_hosp[-1] * (1 - config['div_filter_frac'])
            ub_hosp = real_hosp[-1] * (1 + config['div_filter_frac'])
            profiles = [p for p in profiles]
            # if config['det_history']:
            #     profiles = [p for p in profiles]
            # else:
            #     profiles = [p for p in profiles if lb_hosp < np.sum(p['IH'][last_day_hosp_data]) < ub_hosp or p['seed'] == -1]
            n_replicas = len(profiles)
            T = np.minimum(instance.T, instance.T)  #229
            
            if np.size(instance.epi._gamma_IH) == 25:
                # new specific
                file_type = "new_Specific"
                type_ind = 2
                columnNo = capList.index(instance.hosp_beds)
                rowNo = transmission_file_list.index(str(instance.transmission_file).split("/")[-1])
            elif instance.epi._gamma_IH.det_val == 7.9:
                # new ind
                file_type = "new_Ind"
                type_ind = 1
                columnNo = capList.index(instance.hosp_beds)
                rowNo = transmission_file_list.index(str(instance.transmission_file).split("/")[-1])
            else:
                # old params
                file_type = "old"
                type_ind = 0
                if instance.hosp_beds == 1105:
                    instance.hosp_beds = 1100
                columnNo = capList.index(instance.hosp_beds)
                rowNo = transmission_file_list.index(str(instance.transmission_file).split("/")[-1])
        
            # obtain the statistics
            ICUList = []
            for p in profiles:
                ICUList.append(ratioList[columnNo]*np.max(np.sum(p['IH'],axis = (1,2))))
            perc_results_331[type_ind,rowNo,columnNo] = np.sum(np.array(ICUList) > 331)/300
            perc_results_475[type_ind,rowNo,columnNo] = np.sum(np.array(ICUList) > 475)/300

            ninetyfiveq_results[type_ind,rowNo,columnNo] = np.quantile(ICUList,0.95)
            
            # rename the original file
            instance_name = "austin_{}_{}_{}".format(file_type,instance.hosp_beds,rowNo)
            os.rename(file_path, r"/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/output/ICU_Fixed/" + instance_name + ".p")
            
            # draw the figure
            IHD_plot = plot_multi_tier_sims(instance_name,
                                        instance,
                                        best_policy,
                                        profiles, ['sim'] * len(profiles),
                                        real_hosp,
                                        plot_left_axis=['IH'],
                                        plot_right_axis=[],
                                        T=T,
                                        interventions=interventions,
                                        show=True,
                                        align_axes=True,
                                        plot_triggers=False,
                                        plot_trigger_annotations=False,
                                        plot_legend=False,
                                        y_lim=600,
                                        policy_params=best_params,
                                        n_replicas=n_replicas,
                                        config=config,
                                        hosp_beds_list=[331,475],
                                        bed_scale=ratioList[columnNo]
                                        )
        except:
            pass
        # IYIH_plot = plot_multi_tier_sims(instance_name,
        #                                   instance,
        #                                   best_policy,
        #                                   profiles, ['sim'] * len(profiles),
        #                                   real_hosp,
        #                                   plot_left_axis=['IYIH'],
        #                                   plot_right_axis=[],
        #                                   T=T,
        #                                   interventions=interventions,
        #                                   show=True,
        #                                   align_axes=False,
        #                                   plot_triggers=True,
        #                                   plot_trigger_annotations=False,
        #                                   plot_legend=False,
        #                                   y_lim=None,
        #                                   policy_params=best_params,
        #                                   n_replicas=n_replicas,
        #                                   config=config,
        #                                   hosp_beds_list=hosp_beds_list,
        #                                   real_new_admission=hosp_ad)
        
#%%
folder_name = "Benchmarks"
fileList = os.listdir("/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/output/{}".format(folder_name))

# load Austin real hospitalization
file_path = "/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/instances/austin/austin_real_hosp_updated.csv"
start_date = dt(2020,2,28)
real_hosp = read_hosp(file_path, start_date)
hosp_beds_list = None
file_path = "/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/instances/austin/austin_hosp_ad_updated.csv"
hosp_ad = read_hosp(file_path, start_date, "admits")
file_path = "/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/instances/austin/austin_real_icu_updated.csv"
real_icu = read_hosp(file_path, start_date)

fi = open("/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/output/ICU_Analysis_Austin_{}.csv".format(folder_name),"w",newline="")
csvWriter = csv.writer(fi,dialect='excel')
csvWriter.writerow(['Trigger_Type','CC_Type','Capacity_Cost?','transmission_file','Blue','Yellow','Orange','Red',\
                    '#Red Days','Expected ICU Unmet','No of Scenarios ICU Unmet', 'Largest ICU', '95 Quantile Peak ICU',
                    'Expected IHT Unmet','No of Scenarios IHT Unmet', 'Largest IHT','95 Quantile Peak IHT', 'Deaths','Deaths Stdev'])

for instance_raw in fileList:
    if ".p" in instance_raw and "austin" in instance_raw:
        try:
            instance_name = instance_raw[:-2]
            file_path = f'/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/output/{folder_name}/{instance_name}.p'
            with open(file_path, 'rb') as outfile:
                read_output = pickle.load(outfile)
            instance, interventions, best_params, best_policy, best_sim, profiles, config, cost_record, seeds_info = read_output
            
            # Get only desired profiles
            if real_hosp is None:
                real_hosp = instance.cal.real_hosp
            last_day_hosp_data = len(real_hosp) - 1
            lb_hosp = real_hosp[-1] * (1 - config['div_filter_frac'])
            ub_hosp = real_hosp[-1] * (1 + config['div_filter_frac'])
            profiles = [p for p in profiles]
            # if config['det_history']:
            #     profiles = [p for p in profiles]
            # else:
            #     profiles = [p for p in profiles if lb_hosp < np.sum(p['IH'][last_day_hosp_data]) < ub_hosp or p['seed'] == -1]
            n_replicas = len(profiles)
            T = np.minimum(instance.T, instance.T)  #229
            
            data = []
            # trigger type
            if 'test_IHT_' in instance_name:
                data.append('ToIHT')
            else:
                data.append('ToICU')
            # CC type
            data.append(config['infeasible_field'])
            # over capacity cost included?
            data.append(config['obj_over_included'])
            # transmission file
            data.append(instance.transmission_file)
            # triggers
            for i in range(1,5):
                data.append(best_policy.lockdown_thresholds[i][0])
            # number of red days
            data.append(np.mean([np.sum(profiles[i]['tier_history'][193:] == 4) for i in range(300)]))
            # ICU unmet 
            data.append(np.mean([np.sum(np.maximum(np.sum(profiles[i]['ICU'],axis = (1,2)) - 331,0)) for i in range(300)]))
            data.append(np.sum([np.max(np.sum(profiles[i]['ICU'],axis = (1,2))) > 331 for i in range(300)]))
            data.append(np.max([np.max(np.sum(profiles[i]['ICU'],axis = (1,2))) for i in range(300)]))
            data.append(np.quantile([np.max(np.sum(profiles[i]['ICU'],axis = (1,2))) for i in range(300)], 0.95))

            # IHT unmet 
            data.append(np.mean([np.sum(np.maximum(np.sum(profiles[i]['IHT'],axis = (1,2)) - 1500,0)) for i in range(300)]))
            data.append(np.sum([np.max(np.sum(profiles[i]['IHT'],axis = (1,2))) > 1500 for i in range(300)]))
            data.append(np.max([np.max(np.sum(profiles[i]['IHT'],axis = (1,2))) for i in range(300)]))
            data.append(np.quantile([np.max(np.sum(profiles[i]['IHT'],axis = (1,2))) for i in range(300)], 0.95))

            # deaths
            data.append(np.mean([np.sum(profiles[i]['D'],axis = (1,2))[-1] - np.sum(profiles[i]['D'],axis = (1,2))[193] for i in range(300)]))
            data.append(np.std([np.sum(profiles[i]['D'],axis = (1,2))[-1] - np.sum(profiles[i]['D'],axis = (1,2))[193] for i in range(300)]))
            
            csvWriter.writerow(data)
            
        except:
            pass

fi.close()

#%%
folder_name = "Benchmarks"
fileList = os.listdir("/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/output/{}".format(folder_name))

# load Austin real hospitalization
file_path = "/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/instances/houston/houston_real_hosp_updated.csv"
start_date = dt(2020,2,19)
real_hosp = read_hosp(file_path, start_date)
hosp_beds_list = None
file_path = "/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/instances/houston/houston_real_icu_updated.csv"
real_icu = read_hosp(file_path, start_date)

fi = open("/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/output/ICU_Analysis_Houston_{}.csv".format(folder_name),"w",newline="")
csvWriter = csv.writer(fi,dialect='excel')
csvWriter.writerow(['Trigger_Type','CC_Type','Capacity_Cost?','transmission_file','Blue','Yellow','Orange','Red',\
                    '#Red Days','Expected ICU Unmet','No of Scenarios ICU Unmet', 'Largest ICU', '95 Quantile Peak ICU',
                    'Expected IHT Unmet','No of Scenarios IHT Unmet', 'Largest IHT','95 Quantile Peak IHT', 'Deaths','Deaths Stdev'])

for instance_raw in fileList:
    if ".p" in instance_raw and "houston" in instance_raw:
        try:
            instance_name = instance_raw[:-2]
            file_path = f'/Users/haoxiangyang/Desktop/Git/COVID19_CAOE/InterventionsMIP/output/{folder_name}/{instance_name}.p'
            with open(file_path, 'rb') as outfile:
                read_output = pickle.load(outfile)
            instance, interventions, best_params, best_policy, best_sim, profiles, config, cost_record, seeds_info = read_output
            
            # Get only desired profiles
            if real_hosp is None:
                real_hosp = instance.cal.real_hosp
            last_day_hosp_data = len(real_hosp) - 1
            lb_hosp = real_hosp[-1] * (1 - config['div_filter_frac'])
            ub_hosp = real_hosp[-1] * (1 + config['div_filter_frac'])
            profiles = [p for p in profiles]
            # if config['det_history']:
            #     profiles = [p for p in profiles]
            # else:
            #     profiles = [p for p in profiles if lb_hosp < np.sum(p['IH'][last_day_hosp_data]) < ub_hosp or p['seed'] == -1]
            n_replicas = len(profiles)
            T = np.minimum(instance.T, instance.T)  #229
            
            data = []
            # trigger type
            if 'test_IHT_' in instance_name:
                data.append('ToIHT')
            else:
                data.append('ToICU')
            # CC type
            data.append(config['infeasible_field'])
            # over capacity cost included?
            data.append(config['obj_over_included'])
            # transmission file
            data.append(instance.transmission_file)

            # triggers
            for i in range(1,5):
                data.append(best_policy.lockdown_thresholds[i][0])
            # number of red days
            data.append(np.mean([np.sum(profiles[i]['tier_history'][203:] == 4) for i in range(300)]))
            # ICU unmet 
            data.append(np.mean([np.sum(np.maximum(np.sum(profiles[i]['ICU'],axis = (1,2)) - 1250,0)) for i in range(300)]))
            data.append(np.sum([np.max(np.sum(profiles[i]['ICU'],axis = (1,2))) > 1250 for i in range(300)]))
            data.append(np.max([np.max(np.sum(profiles[i]['ICU'],axis = (1,2))) for i in range(300)]))
            data.append(np.quantile([np.max(np.sum(profiles[i]['ICU'],axis = (1,2))) for i in range(300)], 0.95))

            # IHT unmet 
            data.append(np.mean([np.sum(np.maximum(np.sum(profiles[i]['IHT'],axis = (1,2)) - 4500,0)) for i in range(300)]))
            data.append(np.sum([np.max(np.sum(profiles[i]['IHT'],axis = (1,2))) > 4500 for i in range(300)]))
            data.append(np.max([np.max(np.sum(profiles[i]['IHT'],axis = (1,2))) for i in range(300)]))
            data.append(np.quantile([np.max(np.sum(profiles[i]['IHT'],axis = (1,2))) for i in range(300)], 0.95))
            
            # deaths
            data.append(np.mean([np.sum(profiles[i]['D'],axis = (1,2))[-1] - np.sum(profiles[i]['D'],axis = (1,2))[203] for i in range(300)]))
            data.append(np.std([np.sum(profiles[i]['D'],axis = (1,2))[-1] - np.sum(profiles[i]['D'],axis = (1,2))[203] for i in range(300)]))

            csvWriter.writerow(data)
            
        except:
            pass

fi.close()