import pandas as pd
import numpy as np
from scipy import rand
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from PyEMD import EEMD
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch
from sklearn.preprocessing import MinMaxScaler
import os
import random
import joblib
os.environ["CUDA_LAUNCH_BLOCKING"]= "1"
np.random.seed(123)
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
torch.backends.cudnn.benchmark= False
torch.backends.cudnn.deterministic= True
plt.rcParams["font.family"]= "Times New Roman"

def bacon_watts_knee_point(x, alpha0, alpha1, alpha2, x1):
    ### from paper Identification and machine learning prediction of knee-point and knee-onset in capacity degradation curves of lithium-ion cells equastion 1.
    return alpha0 + alpha1*(x - x1) + alpha2*(x - x1)*np.tanh((x - x1) / 1e-8)

def double_bacon_watts_model(x, alpha0, alpha1, alpha2, alpha3, x0, x2):
    return alpha0 + alpha1*(x - x0) + alpha2*(x - x0)*np.tanh((x - x0)/1e-8) + alpha3*(x - x2)*np.tanh((x - x2)/1e-8)

def time_sequence_silding_window(x, look_back):
    '''
    the input x sequence data would be modified by a sliding window method for train_x and train_y generation
    '''
    n= len(x)
    x_sliding_window_values= []; y_sliding_window_values= []
    for i in range(0 , n- look_back):
        x_sliding_window_values.append(x[i: look_back+ i])
        y_sliding_window_values.append(x[look_back+ i])
    
    x_sliding_window_values= np.array(x_sliding_window_values).reshape(-1, look_back)
    y_sliding_window_values= np.array(y_sliding_window_values)

    return x_sliding_window_values, y_sliding_window_values

def eemd_decomposition_optimization(signal):
    eemd= EEMD()
    eemd.noise_seed(123)
    
    x= np.arange(1, len(signal)+ 1)
    y= signal
    ### avaliable_max_num_imfs_and_residual --> num of maximun imf could get
    avaliable_max_num_imfs_and_residual= eemd.eemd(y, x, max_imf= -1).shape[0]- 1
    
    del eemd
    r_squared_optimization= None
    r_squared_list= []; x_list= []
    for i_imf in range(1, avaliable_max_num_imfs_and_residual+ 1):
        eemd= EEMD()
        eemd.noise_seed(123)
        e_IMFs= eemd.eemd(y, x, max_imf= i_imf).T
        reconstruction_signal= np.sum(e_IMFs, axis= 1)
        
        SSR= np.sum((reconstruction_signal- np.mean(y))** 2)
        SSTO= np.sum((y- np.mean(y))** 2)
        
        r_squared= SSR/ SSTO
        if r_squared_optimization == None:
            r_squared_optimization= r_squared
        elif r_squared_optimization <= r_squared:
            r_squared_optimization= r_squared
        else:
            print("---------------------------------------------------------------------")
            print("[EEMD param] optimal IMFs number: %2d // optimal r2: %.4f"%(i_imf- 1, r_squared_optimization))
            break

    del eemd
    return i_imf- 1

def ML_model_online_prediction(train_x, train_y, valid_len, trained_model):
    x_sp= np.delete(np.append(train_x[-1, :].ravel(), train_y[-1]), 0)

    pred_valid= []
    pred_sliding_x= x_sp.copy()
    for i in range(0, valid_len):
        pred_y= trained_model.predict(pred_sliding_x.reshape(1, -1)).ravel()
        pred_sliding_x= np.delete(np.append(pred_sliding_x.ravel(), pred_y[0]), 0)

        pred_valid.append(pred_y[0])

    pred_valid= np.array(pred_valid)
    return pred_valid.ravel()

def DL_model_online_prediction(generated_cycle, X, y, model, valid_len):
    '''
    the function for online rolling prediction
    *detailed description: erichou22110957@gmail.com
    '''
    n, seq_len= X.shape
    model.eval()
    pred_valid= []

    X= X.reshape(-1, seq_len, 1)
    update_x_original= X[-1, :, :].T
    
    with torch.no_grad():
        for i in range(0, valid_len):
            cycle_x= float(generated_cycle[i])
            
            if i == 0:
                update_x= np.delete(np.append(X[-1, :, :], y[-1]), 0)
                update_x= np.delete(np.insert(update_x, 1, cycle_x), 0)
                
                update_x= update_x.reshape(1, seq_len, 1)
                prediction_x= model(torch.FloatTensor(update_x).to(device)).to(device)
                prediction_x= prediction_x.cpu().data.numpy()[0]
            
            else:
                update_x= update_x.reshape(1, seq_len, 1)
                prediction_x= model(torch.FloatTensor(update_x).to(device)).to(device)
                prediction_x= prediction_x.cpu().data.numpy()[0]
            
            update_x= np.delete(np.append(update_x_original[-1], prediction_x), 0)
            update_x= np.delete(np.insert(update_x, 1, cycle_x), 0)
            update_x= update_x.reshape(1, -1)
            update_x_original= np.append(update_x_original, update_x, axis= 0)
            
            pred_valid.append(prediction_x[0])

    pred_valid= np.array(pred_valid) 
    return pred_valid

class BI_LSTM_ATTENTION(nn.Module):
    def __init__(self, hidden_size, droprate, num_layers, device):
        super(BI_LSTM_ATTENTION, self).__init__()
        self.hidden_size = hidden_size
        self.droprate= droprate
        self.num_layers = num_layers
        self.recurrent_droprate= droprate
        self.device= device
        
        if self.num_layers == 2:
            self.bi_lstm = nn.LSTM(input_size= hidden_size[0], hidden_size= hidden_size[1], num_layers= num_layers, batch_first= True, bidirectional= True, dropout= self.recurrent_droprate)
            self.Linear_01 = nn.Linear(hidden_size[1]* 2, 1)
        
        else:
            self.bi_lstm = nn.LSTM(input_size= hidden_size[0], hidden_size= hidden_size[1], num_layers= num_layers, batch_first= True, bidirectional= False)
            self.Linear_01 = nn.Linear(hidden_size[1]* 1, 1)
    
    def attention(self, value, quary):
        ### there are two hidden state in bi-lstm layer
        ### concatinate by dim-1 (each batch)
        scale= quary.size(-1)** 0.5

        hidden= torch.cat((quary[0], quary[1]), dim= 1).unsqueeze(2)
        
        attention_score= torch.functional.F.softmax(torch.bmm(value, hidden).squeeze(2)/ scale, 1)
        value= torch.bmm(value.transpose(1, 2), attention_score.unsqueeze(2)).squeeze(2)
        
        return value, attention_score
        
    def forward(self, x):
        if self.num_layers == 2:
            h0= torch.rand(size=(self.num_layers * 2, x.size(0), self.hidden_size[1])).to(self.device)
            c0= torch.rand(size=(self.num_layers * 2, x.size(0), self.hidden_size[1])).to(self.device)
        else:
            h0= torch.rand(size=(self.num_layers * 1, x.size(0), self.hidden_size[1])).to(self.device)
            c0= torch.rand(size=(self.num_layers * 1, x.size(0), self.hidden_size[1])).to(self.device)
            
        out, (h_n, c_n) = self.bi_lstm(x, (h0, c0))
        out, attention_score= self.attention(out, h_n)
        
        out= self.Linear_01(out)
        return out

def rul_prediction(generated_cycle, ML_train_x, ML_train_y, ML_normalizer_x, ML_normalizer_y, DL_train_x, DL_train_y, ML_model, DL_model, eol_value, actual_eol, sp_cycle):
        ML_train_x= ML_normalizer_x.transform(ML_train_x)

        x_sp= np.delete(np.append(ML_train_x[-1, :].ravel(), ML_train_y[-1]), 0)

        n, seq_len= DL_train_x.shape
        DL_train_x= DL_train_x.reshape(-1, seq_len, 1)
        update_x_original= DL_train_x[-1, :, :].T

        ML_pred_valid= []; DL_pred_valid= []; composed_SOH_valid= []

        pred_sliding_x= x_sp.copy()

        DL_model.eval(); i= 0
        while True:
            ML_pred_y= ML_model.predict(pred_sliding_x.reshape(1, -1)).ravel()
            pred_sliding_x= np.delete(np.append(pred_sliding_x.ravel(), ML_pred_y[0]), 0)

            ML_pred_y_inv= ML_normalizer_y.inverse_transform(np.array(ML_pred_y[0]).reshape(-1, 1)).ravel()
            ML_pred_valid.append(ML_pred_y_inv)

            cycle_x= float(generated_cycle[i])
            with torch.no_grad():
                if i == 0:
                    update_x= np.delete(np.append(DL_train_x[-1, :, :], DL_train_y[-1]), 0)
                    update_x= np.delete(np.insert(update_x, 1, cycle_x), 0)
                    
                    update_x= update_x.reshape(1, seq_len, 1)
                    DL_prediction_x= DL_model(torch.FloatTensor(update_x).to(device)).to(device)
                    DL_prediction_x= DL_prediction_x.cpu().data.numpy()[0]
                    
                else:
                    update_x= update_x.reshape(1, seq_len, 1)
                    DL_prediction_x= DL_model(torch.FloatTensor(update_x).to(device)).to(device)
                    DL_prediction_x= DL_prediction_x.cpu().data.numpy()[0]
                
                update_x= np.delete(np.append(update_x_original[-1], DL_prediction_x), 0)
                update_x= np.delete(np.insert(update_x, 1, cycle_x), 0)
                update_x= update_x.reshape(1, -1)
                update_x_original= np.append(update_x_original, update_x, axis= 0)
            
            DL_pred_y_inv= DL_prediction_x[0]
            DL_pred_valid.append(DL_pred_y_inv)
            i+= 1
            composed_SOH= DL_pred_y_inv+ ML_pred_y_inv

            if composed_SOH<= eol_value:
                composed_SOH_valid.append(composed_SOH)
                break
            elif i== len(generated_cycle)- 1:
                composed_SOH_valid.append(composed_SOH)
                break
            else:
                composed_SOH_valid.append(composed_SOH)

        try:
            pred_rul= len(composed_SOH_valid)
            actual_rul= actual_eol- (sp_cycle+ 1)

            predicted_eol= sp_cycle+ 1+ pred_rul
            actual_eol= actual_eol

            print("--------> actu RUL: %4d, actu EOL: %4d"%(actual_rul, actual_eol))
            print("--------> pred RUL: %4d, pred EOL: %4d"%(pred_rul, predicted_eol))
            print("===============================================================")
            print("--------> Err  RUL: %4d, Err  EOL: %4d"%(actual_rul- pred_rul, actual_eol- predicted_eol), "\n\n\n")
            return np.array(composed_SOH_valid).ravel(), generated_cycle[: i]/ 0.0004, pred_rul, predicted_eol

        except Exception:
            raise ("* Didn't reach eol point (0.8) \n"+ "* Change the hidden size or batch size to find RUL again !!")

if __name__== "__main__":
    device= "cuda" if torch.cuda.is_available() else "cpu"
    
    mdoel_weight_saving_path=  r"C:\Users\USER\Desktop\battery_sp_2"

    file_path= r"C:\Users\USER\Desktop\battery_sp_2\Battery_3-_4.csv"

    with open(file_path, "r") as file:
        df= pd.read_csv(file, index_col= False)

    df_Battery_4= df[["Cycle", "#5_SOH"]].dropna()
    df_Battery_4= df_Battery_4[df_Battery_4["#5_SOH"]>= 0.800]

    cycle= np.array(df_Battery_4["Cycle"])

    x_data= np.arange(1, len(df_Battery_4)+ 1)
    y_data= np.array(df_Battery_4["#5_SOH"])

    parameter_bound= [1, -1e-4, -1e-4, len(y_data)* 0.7]
    coe, cov= curve_fit(bacon_watts_knee_point, x_data, y_data, parameter_bound)
    opt_alpha0, opt_alpha1, opt_alpha2, opt_x1= coe
    upper_x1, lower_x1= coe[3]+ 1.96* np.diag(cov)[3], coe[3]- 1.96* np.diag(cov)[3]

    parameter_bound= [opt_alpha0, opt_alpha1 + opt_alpha2/2, opt_alpha2, opt_alpha2/2, 0.8*opt_x1, 1.1*opt_x1]
    coe, cov= curve_fit(double_bacon_watts_model, x_data, y_data, parameter_bound)
    opt_alpha0, opt_alpha1, opt_alpha2, opt_alpha3, opt_x0, opt_x2= coe
    upper_x1, lower_x1= coe[4]+ 1.96* np.diag(cov)[4], coe[4]- 1.96* np.diag(cov)[4]

    ############### knee-onset (double bacon-watts model) ###############
    knee_onset_double_bacon_watts= round(opt_x0)
    print("The optimal knee onset (double Bacon Watts) for battery 2 locate at #", knee_onset_double_bacon_watts, "cycle")

    ################################ EEMD #################################
    max_num_imfs= eemd_decomposition_optimization(y_data)

    eemd= EEMD()
    eemd.noise_seed(123)
    E_IMFs= eemd.eemd(y_data, x_data, max_imf= max_num_imfs)

    print("---------------------------------------------------------------------")
    plt.figure(num= 2, figsize= (20, 10))
    plt.subplot(int(E_IMFs.shape[0]/ 2+ 1), 2+ 1, 1)
    plt.title("Original degradation curve", fontsize= 20)
    plt.plot(x_data, y_data, color= "red")
    plt.grid(True)
    for n in range(0, E_IMFs.shape[0]):
        plt.subplot(int(E_IMFs.shape[0]/ 2+ 1), 2+ 1, n+ 2)
        plt.grid(True)
        
        if n== (E_IMFs.shape[0]- 1):
            plt.title("Residual", fontsize= 20)
            plt.plot(x_data, E_IMFs[n], color= "darkblue")
        else:
            plt.title("IMF"+ str(n+ 1), fontsize= 20)
            plt.plot(x_data, E_IMFs[n], color= "darkblue")
            
    plt.tight_layout()
    plt.show()

    IMFs, Residual= E_IMFs.T[:, : -1], E_IMFs.T[:, -1]

    ####################### train & test split ###########################
    ########### Sp replace to knee onset (double bacon watts) ############
    sp= 1450
    Training_Cycle_Index= [knee_onset_double_bacon_watts, sp]
    Testing_Cycle_Index=  [sp+ 1, -1]

    train_x= x_data[Training_Cycle_Index[0]: Training_Cycle_Index[1]]
    train_y= y_data[Training_Cycle_Index[0]: Training_Cycle_Index[1]]
    
    train_IMFs= IMFs[Training_Cycle_Index[0]: Training_Cycle_Index[1], :]
    train_Residual= Residual[Training_Cycle_Index[0]: Training_Cycle_Index[1]]
    
    train_IMFs= pd.DataFrame(train_IMFs).to_dict("list")
    train_IMFs_1= np.array(train_IMFs[0])
    
    #################### test_y: actual test SOH value ####################
    test_y= y_data[Testing_Cycle_Index[0]: Testing_Cycle_Index[1]]
    
    ################# test_IMFs: actual test IMFs value ###################
    test_IMFs= IMFs[Testing_Cycle_Index[0]: Testing_Cycle_Index[1], :]

    ############## test_Residual: actual test residual value ##############
    test_Residual= Residual[Testing_Cycle_Index[0]: Testing_Cycle_Index[1]]
    
    test_IMFs= pd.DataFrame(test_IMFs).to_dict("list")
    test_IMFs_1= np.array(test_IMFs[0])

    print("---------------------------------------------------------------------")
    plt.figure(num= 3, figsize= (20, 5))
    plt.subplot(int(len(train_IMFs.keys())/ 2+ 1), 2+ 1, 1)
    plt.title("Training set of degradation curve", fontsize= 20)
    plt.plot(train_x, train_y, color= "red")
    plt.grid(True)
    for n in range(0, len(train_IMFs.keys())+ 1):
        plt.subplot(int(len(train_IMFs.keys())/ 2+ 1), 2+ 1, n+ 2)
        plt.grid(True)

        if n== (len(train_IMFs.keys())):
            plt.title("Residual", fontsize= 20)
            plt.plot(train_x, train_Residual, color= "darkblue")
        else:
            plt.title("IMF"+ str(n+ 1), fontsize= 20)
            plt.plot(train_x, train_IMFs[n], color= "darkblue")
            
    plt.tight_layout()
    plt.show()
    
    ML_seq_len= 13
    ### Sliding Window For Generate Data Sequence ##################################################
    x_train_IMFs, y_train_IMFs= time_sequence_silding_window(x= train_IMFs_1, look_back= ML_seq_len)
    ### SVR Model Prediction ######################################################################
    
    pre_traned_ML_model= joblib.load(os.path.join(mdoel_weight_saving_path, "SVR_model"))
    ML_normalizer_x= joblib.load(os.path.join(mdoel_weight_saving_path, "Normalizer_x_model"))
    ML_normalizer_y= joblib.load(os.path.join(mdoel_weight_saving_path, "Normalizer_y_model"))

    x_train_IMFs= ML_normalizer_x.transform(x_train_IMFs)
    online_prediction_valid_y= ML_model_online_prediction(x_train_IMFs, y_train_IMFs, len(test_IMFs_1), trained_model= pre_traned_ML_model)

    pred_valid= ML_normalizer_y.inverse_transform(online_prediction_valid_y.reshape(-1, 1)).ravel()

    model_libaray= r"C:\Users\USER\Desktop\battery_sp_2\model_check_point"
    DL_model_check_point_list= os.listdir(model_libaray)

    cycle= cycle* 0.0004
    cycle= cycle[Training_Cycle_Index[0]: Training_Cycle_Index[1]]

    selected_param= {"seq_len": 31, "hidden_size": 105, "batch_size": 71}

    seq_len_param= int(selected_param["seq_len"])
    hidden_size_param= int(selected_param["hidden_size"])
    batch_size_param= int(selected_param["batch_size"])
    epoch_param= 1000

    check_point= "residual_model_seq_len_"+ str(seq_len_param)+ "_hidden_size_"+ str(hidden_size_param)+ "_batch_size_"+ str(batch_size_param)

    x_train_Residual_in, y_train_Residual_in= time_sequence_silding_window(x= train_Residual, look_back= seq_len_param)
    x_train_Residual_in= np.append(cycle[seq_len_param: ].reshape(-1, 1), x_train_Residual_in, axis= 1)

    pre_traned_DL_model= torch.load(os.path.join(mdoel_weight_saving_path, "model_check_point", check_point, "residual_model.pth"))

    generated_cycle = np.arange(x_train_Residual_in[-1, :].ravel()[0] + 0.0004, 2.0000, 0.0004)
    pred_valid= DL_model_online_prediction(generated_cycle= generated_cycle, X= x_train_Residual_in, y= y_train_Residual_in, model= pre_traned_DL_model, valid_len= len(test_Residual))

    actual_eol= len(df_Battery_4)
    actual_eol_soh= 0.8

    composed_SOH, pred_cycle, pred_rul, pred_eol= rul_prediction(generated_cycle= generated_cycle,
                                                                    ML_train_x= x_train_IMFs,  ML_train_y= y_train_IMFs,
                                                                    DL_train_x= x_train_Residual_in, DL_train_y= y_train_Residual_in, 
                                                                    ML_model= pre_traned_ML_model, DL_model= pre_traned_DL_model, 
                                                                    ML_normalizer_x= ML_normalizer_x, ML_normalizer_y= ML_normalizer_y, 
                                                                    eol_value= actual_eol_soh, actual_eol= actual_eol, sp_cycle= Training_Cycle_Index[1])
    
    plt.figure(figsize= (13, 4))
    plt.title("online rul prediction for battery 2.", fontsize= 20, fontweight= "bold")
    plt.plot(np.arange(0, Training_Cycle_Index[1]), y_data[: Training_Cycle_Index[1]], label= "battery_2 (known trend curve)", color= "black")
    plt.plot(np.arange(Testing_Cycle_Index[0]+ 1, len(y_data)), test_y, label= "battery_2 (unknown trend curve)", color= "blue")
    plt.plot(pred_cycle, composed_SOH, label= "SOH online prediction", color= "red", alpha= 0.5)
    plt.scatter(knee_onset_double_bacon_watts, y_data[knee_onset_double_bacon_watts], label= "knee onset (double Bacon Watts)", color= "darkred", s= 50, marker= "o")
    plt.scatter(Training_Cycle_Index[1], y_data[Training_Cycle_Index[1]], label= "knee onset (sp)", color= "darkred", s= 50, marker= "o")
    plt.hlines(0.8, 0, actual_eol, color= "red", label= "EOL (SOH= 0.8)", ls= "--")
    plt.ylabel("SOH", fontsize= 20)
    plt.xlim(0, pred_eol)
    plt.hlines(0.8, 0, pred_eol, color= "red", label= "EOL (SOH= 0.8)", ls= "--")
    plt.xlabel("cycle life", fontsize= 20)
    plt.legend(loc= "best")
    plt.show()

    # print("--------> actu RUL:%4d, actu EOL:%4d"%(actual_eol- Training_Cycle_Index[1], actual_eol))
    # print("--------> pred RUL:%4d, pred EOL:%4d"%(round(np.mean(est_rul), 4), round(np.mean(est_eol), 4)))
    # print("===============================================================")
    # print("--------> Erro RUL:%4d, Erro EOL: %4d"%(round(actual_eol- Training_Cycle_Index[1]- np.mean(est_rul), 4), round(actual_eol- np.mean(est_eol), 4)), "\n\n\n")