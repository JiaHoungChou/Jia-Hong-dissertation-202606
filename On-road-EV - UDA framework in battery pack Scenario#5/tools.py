import os, torch, copy, warnings
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from dtaidistance import dtw
from xlstm import sLSTM
from scipy.optimize import curve_fit
import matplotlib
plt.rcParams["font.family"] = "Times New Roman"
warnings.filterwarnings("ignore")
# matplotlib.rcParams.update({"text.usetex": True,"font.family": "serif", "font.serif": ["Times New Roman"], "text.latex.preamble": r"\usepackage{mathptmx}"})
device= "cuda" if torch.cuda.is_available else "cpu"

def real_csvfiles_in_folder(folder_path):
    battery_data_ls= [file_name for file_name in os.listdir(folder_path)]
    dataframe= pd.DataFrame()
    for file in battery_data_ls:
        file_path= os.path.join(folder_path, file)
        data= pd.read_csv(file_path)
        dataframe= pd.concat([dataframe, data], axis= 0)
    dataframe= dataframe.reset_index(drop= True)
    return dataframe

def bacon_watts_knee_point(x, alpha0, alpha1, alpha2, x1, scaler):
    scaler= -1* scaler
    return alpha0+ alpha1*(x - x1) + scaler* alpha2*(x - x1)* np.tanh((x - x1)/ 1e-8)

def func(x, a, b, c):
    return -1* a* np.exp(b*0.05*  -1* x)+ c

def get_Tsp(source_dataset: pd.DataFrame, verbose= False):
    knee_point_ls= []; Tsp_capacity_ls= []
    for cell_name in np.unique(source_dataset["battery_name"]):
        cell_curve= source_dataset[source_dataset["battery_name"]== cell_name]
        cell_curve_fit_knee_point= cell_curve[cell_curve["charge_capacity"]>= 0.86]
        x_data= np.arange(1, len(cell_curve_fit_knee_point["charge_capacity"])+ 1)
        y_data= np.array(cell_curve_fit_knee_point["charge_capacity"])
        parameter_bound= [1, -1e-4, -1e-4, len(cell_curve), 5]
        popt, _= curve_fit(func, x_data, y_data, maxfev=10000000)
        coe, _= curve_fit(bacon_watts_knee_point, x_data, func(x_data, *popt), parameter_bound)
        opt_alpha0, opt_alpha1, opt_alpha2, opt_x1, scaler= coe
        knee_point= round(opt_x1)
        if knee_point<= 0 or knee_point>= len(y_data):
            continue
        else:
            if verbose== True:
                plt.plot(x_data, y_data* 3500, color= "darkblue", label= "capacity curve")
                plt.plot(x_data, bacon_watts_knee_point(x_data, opt_alpha0, opt_alpha1, opt_alpha2, opt_x1, scaler)* 3500, color= "darkred", label= "Bacon-Watts model")
                plt.scatter(knee_point, y_data[knee_point]* 3500, label= "knee point: %s cycle"%(str(knee_point)), color= "black", s= 50, marker= "o")
                plt.legend(loc= "best", fontsize= 15)
                plt.xlabel("Cycles", fontsize= 15)
                plt.ylabel("Capacity (mAh)", fontsize= 15)
                plt.xticks(fontsize= 15)
                plt.yticks(fontsize= 15)
                plt.show()
            knee_point_ls.append(knee_point)
            Tsp_capacity_ls.append(y_data[knee_point])
    return round(np.mean(knee_point_ls)), round(np.mean(Tsp_capacity_ls), 4)

def data_smoothing(dataset: pd.DataFrame, lowess_tau: float, verbose= False):
    new_dataset= pd.DataFrame()
    if verbose== True:
        if len(np.unique(dataset["battery_name"]))> 10:
            plt.figure(figsize= (10, 6)); index= 1
        elif len(np.unique(dataset["battery_name"]))>= 5:
            plt.figure(figsize= (10, 6)); index= 1
        else:
            plt.figure(figsize= (8, 6)); index= 1
    for cells in np.unique(dataset["battery_name"]):
        sub_dataset, capacity= dataset[dataset["battery_name"]== cells], dataset[dataset["battery_name"]== cells]["charge_capacity"]
        smoothed_y= sm.nonparametric.lowess(capacity, np.arange(0, len(capacity)), frac= lowess_tau)[:, 1]
        sub_dataset["charge_capacity"]= smoothed_y
        if verbose== True:
            plt.gca().set_title(cells)
            plt.subplot(int(len(np.unique(dataset["battery_name"]))/ 4), 4, index)
            plt.plot(np.arange(0, len(capacity)), capacity, color= "darkblue", alpha= 0.3)
            plt.plot(np.arange(0, len(capacity)), smoothed_y, color= "darkred", ls= "--")
            plt.grid(True)
            plt.xticks(fontsize= 15)
            plt.yticks(fontsize= 15)
            plt.legend(loc= "best", labels= ["Original", "Smoothed"])
            index+= 1
        new_dataset= pd.concat([new_dataset, sub_dataset], axis= 0)
    if verbose== True:
        plt.tight_layout()
        plt.savefig("revision.png", dpi= 700)
        plt.show()
    return new_dataset

def get_figure1(SOURCE_DOMAIN, TARGET_DOMAIN, source_full_cell_dataset, source_partial_cell_dataset, target_partial_cell_dataset, normal_capacity, Tsp_capacity):
    plt.figure(num= 1, figsize= (8, 6)); count= 0; max_len= 0
    plt.subplot(3, 1, 1)
    source_cell_name, target_cell_name= np.unique(source_partial_cell_dataset["Cell"]), np.unique(target_partial_cell_dataset["Cell"])
    for cell_name in source_cell_name:
        cell_dataset= source_full_cell_dataset[source_full_cell_dataset["Cell"]== cell_name]
        cell_capacity= cell_dataset["Capacity"].values
        cell_capacity= cell_capacity[cell_capacity>= normal_capacity* 0.80]
        max_len= len(cell_capacity) if  len(cell_capacity)> max_len else max_len
        if count== 0:
            plt.plot(np.arange(0, len(cell_capacity)), cell_capacity, color= "darkgreen", label= "[Source] %s in temperature %s $^{o}C$ (full)"%(SOURCE_DOMAIN.split("-")[0].upper(), SOURCE_DOMAIN.split("-")[1]))
        else:
            plt.plot(np.arange(0, len(cell_capacity)), cell_capacity, color= "darkgreen")
        count+= 1
    plt.hlines(Tsp_capacity, 0, max_len, color= "red", ls= "--", label= "Psp")
    plt.hlines(normal_capacity* 0.80, 0, max_len, color= "darkred", ls= "--", label= "EOL")
    plt.xlim(0, max_len)
    plt.xlabel("Cycles", fontsize= 15)
    plt.ylabel("Capacity (mAh)", fontsize= 15)
    plt.xticks(fontsize= 15)
    plt.yticks(fontsize= 15)
    plt.legend(loc= "best", fontsize= 10)
    plt.subplot(3, 1, 2)
    count= 0
    for cell_name in source_cell_name:
        cell_dataset= source_partial_cell_dataset[source_partial_cell_dataset["Cell"]== cell_name]
        cell_capacity= cell_dataset["Capacity"].values
        max_len= len(cell_capacity) if  len(cell_capacity)> max_len else max_len
        if count== 0:
            plt.plot(np.arange(0, len(cell_capacity)), cell_capacity, color= "black", label= "[Source] %s in temperature %s $^{o}C$ (partial)"%(SOURCE_DOMAIN.split("-")[0].upper(), SOURCE_DOMAIN.split("-")[1]))
        else:
            plt.plot(np.arange(0, len(cell_capacity)), cell_capacity, color= "black")
        count+= 1
    plt.hlines(Tsp_capacity, 0, max_len, color= "red", ls= "--", label= "Psp")
    plt.hlines(normal_capacity* 0.80, 0, max_len, color= "darkred", ls= "--", label= "EOL")
    plt.xlim(0, max_len)
    plt.xlabel("Cycles", fontsize= 15)
    plt.ylabel("Capacity (mAh)", fontsize= 15)
    plt.xticks(fontsize= 15)
    plt.yticks(fontsize= 15)
    plt.legend(loc= "best", fontsize= 10)
    plt.subplot(3, 1, 3)
    count= 0
    for cell_name in target_cell_name:
        cell_dataset= target_partial_cell_dataset[target_partial_cell_dataset["Cell"]== cell_name]
        cell_capacity= cell_dataset["Capacity"].values
        max_len= len(cell_capacity) if  len(cell_capacity)> max_len else max_len
        if count== 0:
            plt.plot(np.arange(0, len(cell_capacity)), cell_capacity, color= "darkblue", label= '[Target] %s in temperature %s $^{o}C$  (partial)'%(TARGET_DOMAIN.split("-")[0].upper(), TARGET_DOMAIN.split("-")[1]))
        else:
            plt.plot(np.arange(0, len(cell_capacity)), cell_capacity, color= "darkblue")
        count+= 1
    plt.hlines(Tsp_capacity, 0, max_len, color= "red", ls= "--", label= "Psp")
    plt.hlines(normal_capacity* 0.80, 0, max_len, color= "darkred", ls= "--", label= "EOL")
    plt.xlim(0, max_len)
    plt.xlabel("Cycles", fontsize= 15)
    plt.ylabel("Capacity (mAh)", fontsize= 15)
    plt.xticks(fontsize= 15)
    plt.yticks(fontsize= 15)
    plt.legend(loc= "best", fontsize= 10)
    plt.show()
    plt.tight_layout()

def sliding_window_approach(dataset, m):
    cells= np.unique(dataset["battery_name"]); sliding_window_dataset= pd.DataFrame()
    for cell_i in range(0, len(cells)):
        cell_dataset= dataset[dataset["battery_name"]== cells[cell_i]]
        cell_name, cell_capacity= cell_dataset["battery_name"], np.array(cell_dataset["charge_capacity"])
        data_x= np.zeros(shape= (len(cell_capacity)- m, m)); data_y= np.zeros(shape= (len(cell_capacity)- m))
        for c_i in range(0, len(cell_capacity)- m):
            data_x[c_i, :]= cell_capacity[c_i: c_i+ m]; data_y[c_i]= cell_capacity[c_i+ m- 1]- cell_capacity[c_i+ m]
        data_x= pd.DataFrame(data_x); data_y= pd.DataFrame(data_y); Cell_name= pd.DataFrame(cell_name[: len(data_x)]).reset_index(drop= True)
        data_x.columns, data_y.columns, Cell_name.columns= ["capacity%s"%(i+ 1) for i in range(0, m)], ["capacity_fade"], ["cell"]
        sliding_window_dataset= pd.concat([sliding_window_dataset, pd.concat([Cell_name, data_x, data_y], axis= 1)], axis= 0)
    sliding_window_dataset= sliding_window_dataset.reset_index(drop= True)
    return sliding_window_dataset

def data_normalization(dataset: pd.DataFrame):
    normalizer= MinMaxScaler()
    dataset["charge_capacity"]= normalizer.fit_transform(dataset[["charge_capacity"]])
    return dataset, normalizer

def partial_dataset_generator(dataset: pd.DataFrame, Tsp: float):
    cell_ls= np.unique(dataset["battery_name"]); partial_dataset= pd.DataFrame()
    for cell_name in cell_ls:
        cell_data= dataset[dataset["battery_name"]== cell_name]
        cell_data= cell_data[cell_data["charge_capacity"]>= Tsp]
        partial_dataset= pd.concat([partial_dataset, cell_data], axis= 0)
    partial_dataset= partial_dataset.reset_index(drop= True)
    return partial_dataset

def train_dataset_generator(dataset: pd.DataFrame, num_features):
    capacity_columns= [c_i for c_i in dataset.columns if "capacity" in c_i ][: -1]
    lstm_input_dataset= np.zeros(shape= (len(dataset), len(capacity_columns), num_features))
    for index in range(0, len(lstm_input_dataset)):
        capacity_lookback_sequence= np.array(dataset[capacity_columns].iloc[index].values).reshape(-1, 1)
        lstm_input_dataset[index]= capacity_lookback_sequence
    lstm_x, lstm_y= copy.copy(lstm_input_dataset), copy.copy(dataset["capacity_fade"].values)
    return lstm_x, lstm_y

class scalerLSTM(nn.Module):
    def __init__(self, in_dim, head_size, num_heads, model_mode= "train"):
        super(scalerLSTM, self).__init__()
        self.in_dim, self.device= in_dim, device
        self.head_size, self.num_heads= head_size, num_heads
        self.is_train= True if model_mode== "train" else False
        self.lstm= sLSTM(input_size= self.in_dim, head_size= self.head_size, num_heads= self.num_heads, num_layers=1, batch_first= False, proj_factor=4/3)
    def gussain_noise_layer(self, input, mean= 0.0, standard_deviaton= 0.00001):
        if self.is_train:
            noise= input.data.new(input.size()).normal_(mean, standard_deviaton)
            return noise+ input
        return input
    def forward(self, input_x):
        input_x= self.gussain_noise_layer(input_x)
        lstm_output, _= self.lstm(input_x)
        return lstm_output, lstm_output[:, -1, :]
    
class Predictor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layer_size, dropout_rate, is_train= False):
        super(Predictor, self).__init__()
        self.in_dim, self.out_dim, self.hidden_layer_size, self.is_train= in_dim, out_dim, hidden_layer_size, is_train
        self.dense1= nn.Linear(in_dim, self.hidden_layer_size["dense_layer_1"])
        self.dense2= nn.Linear(self.hidden_layer_size["dense_layer_1"], self.hidden_layer_size["dense_layer_2"])
        self.dropout_layer= nn.Dropout(p= dropout_rate)
        self.ReLU= nn.ReLU()
        self.output_layer= nn.Linear(self.hidden_layer_size["dense_layer_2"], out_dim)
    def gussain_noise_layer(self, input, mean= 0.00, standard_deviaton= 0.00001):
        if self.is_train:
            noise= input.data.new(input.size()).normal_(mean, standard_deviaton)
            return noise+ input
        return input
    def forward(self, input_x):
        output= self.ReLU(input_x)
        output= self.dense1(output)
        output= self.dropout_layer(output)
        output= self.ReLU(output)
        output= self.dense2(output)
        output= self.dropout_layer(output)
        output= self.ReLU(output)
        output= self.output_layer(output)
        output= self.gussain_noise_layer(output)
        # output= torch.abs(nn.functional.tanh(output))
        output= nn.functional.tanh(output)
        return output

def _softmin(a, b, c, gamma):
    stacked = torch.stack([a, b, c], dim= -1)
    return -gamma * torch.logsumexp(-stacked / gamma, dim=-1)

class SoftDTWLoss(torch.nn.Module):
    def __init__(self, gamma= 0.1, normalize= True):
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize
    def forward(self, x, y):
        assert x.dim() == 3 and y.dim() == 3, "x,y must be [B,T,D]"
        B, T1, D= x.shape
        _, T2, _= y.shape
        gamma = self.gamma
        Dxy = (x.unsqueeze(2) - y.unsqueeze(1)).pow(2).sum(dim=3)
        inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        R = inf.expand(B, T1 + 1, T2 + 1).clone()
        R[:, 0, 0] = 0.0
        for i in range(1, T1 + 1):
            for j in range(1, T2 + 1):
                r0 = R[:, i - 1, j]
                r1 = R[:, i, j - 1]
                r2 = R[:, i - 1, j - 1]
                R[:, i, j] = Dxy[:, i - 1, j - 1] + _softmin(r0, r1, r2, gamma)
        loss = R[:, T1, T2]
        if self.normalize:
            loss_xx = self._self_dtw(x)
            loss_yy = self._self_dtw(y)
            loss = loss - 0.5 * loss_xx - 0.5 * loss_yy
        return loss.mean()
    def _self_dtw(self, x):
        B, T, D = x.shape
        gamma = self.gamma
        Dxx = (x.unsqueeze(2) - x.unsqueeze(1)).pow(2).sum(dim= 3)
        inf = torch.tensor(float("inf"), device=x.device, dtype=x.dtype)
        R = inf.expand(B, T + 1, T + 1).clone()
        R[:, 0, 0] = 0.0
        for i in range(1, T + 1):
            for j in range(1, T + 1):
                r0 = R[:, i - 1, j]
                r1 = R[:, i, j - 1]
                r2 = R[:, i - 1, j - 1]
                R[:, i, j] = Dxx[:, i - 1, j - 1] + _softmin(r0, r1, r2, gamma)
        return R[:, T, T]

def get_figure(actual, preidction, figure_title):
    plt.figure(figsize= (6, 3))
    plt.title(figure_title, fontsize= 18)
    plt.plot(np.arange(0, len(actual)), actual, color= "darkorange", label= "actual capacity")
    plt.plot(np.arange(0, len(preidction)), preidction, color= "darkblue", label= "predicted capacity")
    plt.legend(loc= "best", fontsize= 15)
    plt.xlabel("cycle", fontsize= 15)
    plt.ylabel("mAh", fontsize= 15)
    plt.xticks(fontsize= 12)
    plt.yticks(fontsize= 12)
    plt.show()

def get_inference_figure(actual, preidction, Tsp, EOL_capacity, figure_title):
    plt.figure(figsize= (6, 3))
    plt.title(figure_title, fontsize= 18)
    plt.plot(np.arange(0, len(actual)), actual, color= "darkorange", label= "actual capacity")
    plt.plot(np.arange(Tsp, Tsp+ len(preidction)), preidction, color= "darkblue", label= "predicted capacity")
    plt.hlines(EOL_capacity, 0, max([len(actual), len(preidction)]), color= "darkred", ls= "--", label= "EOL")
    plt.legend(loc= "best", fontsize= 15)
    plt.xlabel("cycle", fontsize= 15)
    plt.ylabel("mAh", fontsize= 15)
    plt.xticks(fontsize= 12)
    plt.yticks(fontsize= 12)
    plt.show()
