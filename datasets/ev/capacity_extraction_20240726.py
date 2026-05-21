import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapz
import datetime as dt
from natsort import ns, natsorted
from matplotlib import rcParams
params={'font.family': 'serif', 'font.serif': 'Times New Roman', 'font.style': 'normal', 'font.weight':'normal', 'font.size': 12}
rcParams.update(params)

if __name__ == '__main__':
    main_path= os.getcwd()
    dird= os.listdir(main_path)
    dird= natsorted(dird, alg= ns.PATH)
    file_name_ls= [file_name for file_name in dird if file_name.split(".")[-1]== "csv"]
    for veh_file in file_name_ls:
        file_path= os.path.join(main_path, veh_file)
        file= pd.read_csv(file_path)
        file.columns = ['number', 'record_time', 'soc', 'pack_voltage', 'charge_current', 'max_cell_voltage', 'min_cell_voltage', 'max_temperature', 'min_temperature',  'available_energy', 'available_capacity']
        file= file.sort_values(by= 'record_time')
        file.reset_index(drop= True, inplace= True)
        charge_time= []
        for index in range(0, len(file)):
            charge_time.append(str(file['record_time'][index]))
        charge_time= pd.to_datetime(np.array(charge_time))
        charge_time= pd.DataFrame(charge_time)
        Tm0= charge_time.iloc[1: ]
        Tm0.reset_index(drop= True, inplace= True)
        Tm1= charge_time.iloc[: -1]
        Tm1.reset_index(drop= True, inplace= True)
        ### sampling rate (interval: 8 sec).
        time_delta= (Tm0- Tm1)
        sample_extraction_interval= dt.timedelta(seconds=10)
        ### save the second interval larger than 10 sec.
        rest_index= []
        for i in range(0, len(time_delta)):
            if time_delta.iloc[i, 0] > sample_extraction_interval:
                rest_index.append(i)
        charge_list= []
        charge_list.append(file.iloc[: rest_index[0]])
        for index in range(0, len(rest_index)- 1):
            charge_sub= file.iloc[rest_index[index]+1: rest_index[index+ 1]]
            charge_list.append(charge_sub)
        charge_list.append(file.iloc[rest_index[-1]+ 1: ])
        real_capacity_list= []; charge_list_out= []
        for index in range(0, len(charge_list)):
            ### cha_cut (shape -> 640 rows x 11 columns)
            charge_sub= charge_list[index]
            charge_sub.reset_index(drop= True, inplace= True)
            if len(charge_sub)<100:
                continue
            soc_delta= charge_sub['soc'][1: ]- charge_sub['soc'][: -1]
            if np.sum(soc_delta> 2) or np.sum(soc_delta< -0.1):
                continue
            time= []
            for index in range(0, len(charge_sub)):
                time.append(str(charge_sub['record_time'][index]))
            time= np.array(time)
            time= pd.to_datetime(time)
            current, soc, Tmax, Tmin= charge_sub['charge_current'], charge_sub['soc'], np.mean(charge_sub['max_temperature']), np.mean(charge_sub['min_temperature'])
            ### capacity data sorting.
            if np.sum(np.isnan(current.tolist()))> len(current)* 0.1:
                continue
            if np.sum(np.isnan(current.tolist())):
                for n in range(0, len(current)):
                    if np.isnan(current[n]):
                        current[n]= current[n-1]
            time_sec= np.zeros(len(current))
            for index in range(0, len(current)):
                time_temp= time[index]- time[0]
                ### total second.
                time_sec[index]= time_temp.total_seconds()
            ### integral computation.
            accumulated_Q= trapz(current, time_sec)/3600*(-1)
            soc_delta= soc[len(soc)- 1]- soc[0]
            if soc_delta==0:
                continue
            real_capacity= accumulated_Q/ soc_delta* 100
            real_capacity_list.append([time[0], time[len(time)- 1], soc[0], soc[len(time)-1],  real_capacity, Tmax, Tmin])
            charge_list_out.append(charge_sub)
        ### final sorted file save.
        vehicle_dataframe= pd.DataFrame(data= real_capacity_list, columns=['time_s', 'time_e', 'SOC_s', 'SOC_e', 'charge_capacity', 'Tmax', 'Tmin'])
        vehicle_dataframe.reset_index(drop=True, inplace=True)
        vehicle_dataframe.insert(0, "cycle_num", [i+ 1 for i in range(0, len(vehicle_dataframe))])
        # vehicle_dataframe.to_csv(os.path.join(main_path, "preprocessed_ev_batteries", veh_file), index= False)
        cnt=0; time_index=[]
        time_index.append(vehicle_dataframe.time_e[0])
        ca_month=[]; ca_temp=[]
        for index in range(0, len(vehicle_dataframe)):
            if(vehicle_dataframe.time_e[index].year== time_index[cnt].year)and(vehicle_dataframe.time_e[index].month== time_index[cnt].month):
                ca_temp.append(vehicle_dataframe.charge_capacity[index])
            else:
                ca_month.append(ca_temp)
                cnt= cnt+1
                time_index.append(vehicle_dataframe.time_e[index])
                ca_temp=[]
                ca_temp.append(vehicle_dataframe.charge_capacity[index])
        ca_month.append(ca_temp)
        vehicle_capacity_mean= [np.mean(p) for p in ca_month] 
        vehicle_capacity_median= [np.median(p) for p in ca_month] 
        plt.figure(figsize= (6, 3)) 
        plt.plot(time_index, vehicle_capacity_mean)
        plt.plot(time_index, vehicle_capacity_median, '-.')
        plt.xticks(rotation= 90)
        plt.yticks(np.arange(110, 135, 6))
        plt.ylabel('Capacity (Ah)')
        plt.xlabel('Date')
        plt.legend(['Mean', 'Median'],loc=1)
        plt.show()