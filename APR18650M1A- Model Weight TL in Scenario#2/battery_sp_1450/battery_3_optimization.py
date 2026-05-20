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

def ML_model_train_and_valid_processing(gamma: float, C: float, epsilon: float, train_x: np.array, train_y: np.array, valid_y: np.array):
    train_x, train_y, valid_y= train_x, train_y, valid_y

    normalizer_x= MinMaxScaler()
    normalizer_y= MinMaxScaler()

    train_x= normalizer_x.fit_transform(train_x)
    train_y= normalizer_y.fit_transform(train_y.reshape(-1, 1)).ravel()

    model_= SVR(kernel= "rbf", gamma= gamma, C= C, epsilon= epsilon)
    model_.fit(train_x, train_y.ravel())

    pred_valid= ML_model_online_prediction(train_x, train_y, len(valid_y), model_)
    pred_valid= normalizer_y.inverse_transform(pred_valid.reshape(-1, 1)).ravel()

    train_y= normalizer_y.inverse_transform(train_y.reshape(-1, 1)).ravel()
    
    pred_train_y= model_.predict(train_x)
    pred_train_y= normalizer_y.inverse_transform(pred_train_y.reshape(-1, 1)).ravel()

    plt.figure(figsize= (13, 4))
    plt.title("SVR model (history prediction)", fontsize= 15)
    plt.plot(np.arange(0, len(train_y)), train_y, label= "train_y")
    plt.plot(np.arange(0, len(pred_train_y)), pred_train_y, label= "pred_train_y")
    plt.legend(loc= "best")
    plt.show()

    plt.figure(figsize= (13, 4))
    plt.title("SVR model (online prediction)", fontsize= 15)
    plt.plot(np.arange(0, len(valid_y)), valid_y, label= "valid_y")
    plt.plot(np.arange(0, len(pred_valid)), pred_valid, label= "rolling pred_valid")
    plt.legend(loc= "best")
    plt.show()

    rmse= round(np.mean((valid_y- pred_valid)** 2)** (1/ 2), 4)
    return model_, normalizer_x, normalizer_y

class ML_optimziation_process:
    def __init__(self, x_imf_i, test_imf_i):
        self.x_imf_i= x_imf_i
        self.test_imf_i= test_imf_i

    def time_sequence_silding_window(self, x, look_back):
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

    def ML_model_online_prediction(self, train_x, train_y, valid_len, trained_model):
        x_sp= np.delete(np.append(train_x[-1, :].ravel(), train_y[-1]), 0)
        pred_valid= []
        pred_sliding_x= x_sp.copy()
        for i in range(0, valid_len):
            pred_y= trained_model.predict(pred_sliding_x.reshape(1, -1)).ravel()
            pred_sliding_x= np.delete(np.append(pred_sliding_x.ravel(), pred_y[0]), 0)

            pred_valid.append(pred_y[0])

        pred_valid= np.array(pred_valid)
        return pred_valid.ravel()

    def process(self, seq_len: int, gamma: float, C: float, epsilon: float, x: np.array, valid_y: np.array):
        x_train_IMFs, y_train_IMFs= self.time_sequence_silding_window(x= x, look_back= seq_len)

        train_x, train_y, valid_y= x_train_IMFs, y_train_IMFs, valid_y

        normalizer_x= MinMaxScaler()
        normalizer_y= MinMaxScaler()
        
        train_x= normalizer_x.fit_transform(train_x)
        train_y= normalizer_y.fit_transform(train_y.reshape(-1, 1)).ravel()

        model_= SVR(kernel= "rbf", gamma= gamma, C= C, epsilon= epsilon)
        model_.fit(train_x, train_y.ravel())

        pred_valid= ML_model_online_prediction(train_x, train_y, len(valid_y), model_)
        pred_valid= normalizer_y.inverse_transform(pred_valid.reshape(-1, 1)).ravel()

        train_y= normalizer_y.inverse_transform(train_y.reshape(-1, 1)).ravel()
        
        pred_train_y= model_.predict(train_x)
        pred_train_y= normalizer_y.inverse_transform(pred_train_y.reshape(-1, 1)).ravel()

        del model_
        rmse= round(np.mean((valid_y- pred_valid)** 2)** (1/ 2), 4)
        return rmse

    def param_tuning_by_process(self, bounds):
        look_back_param= int(bounds[0])
        gamma_param= float(bounds[1])
        c_param= float(bounds[2])
        epsilon_param= float(bounds[3])
        rmse= self.process(seq_len= look_back_param, gamma= gamma_param, C= c_param, epsilon= epsilon_param, x= self.x_imf_i, valid_y= self.test_imf_i)
        return rmse

    def run_optimization(self, param_bounds, max_iteration, visualization= None):
        self.result= differential_evolution(self.param_tuning_by_process, param_bounds, maxiter= max_iteration)

        opt_look_back, opt_gamma, opt_c, op_epsilon= int(self.result.x[0]), float(self.result.x[1]), float(self.result.x[2]), float(self.result.x[3])
        print("opt_look_back: %3d, opt_gamma: %4.6f, opt_c: %4.6f, opt_eps: %4.6f"%(opt_look_back, opt_gamma, opt_c, op_epsilon))

        x_train_IMFs, y_train_IMFs= self.time_sequence_silding_window(x= self.x_imf_i, look_back= opt_look_back)

        train_x, train_y, valid_y= x_train_IMFs, y_train_IMFs, self.test_imf_i

        normalizer_x= MinMaxScaler()
        normalizer_y= MinMaxScaler()
        
        train_x= normalizer_x.fit_transform(train_x)
        train_y= normalizer_y.fit_transform(train_y.reshape(-1, 1)).ravel()

        model_= SVR(kernel= "rbf", gamma= opt_gamma, C= opt_c, epsilon= op_epsilon)
        model_.fit(train_x, train_y.ravel())

        pred_valid= ML_model_online_prediction(train_x, train_y, len(valid_y), model_)
        pred_valid= normalizer_y.inverse_transform(pred_valid.reshape(-1, 1)).ravel()

        train_y= normalizer_y.inverse_transform(train_y.reshape(-1, 1)).ravel()
        
        pred_train_y= model_.predict(train_x)
        pred_train_y= normalizer_y.inverse_transform(pred_train_y.reshape(-1, 1)).ravel()

        plt.figure(figsize= (13, 4))
        plt.title("SVR model (history prediction)", fontsize= 15)
        plt.plot(np.arange(0, len(train_y)), train_y, label= "train_y")
        plt.plot(np.arange(0, len(pred_train_y)), pred_train_y, label= "pred_train_y")
        plt.legend(loc= "best")
        plt.show()

        plt.figure(figsize= (13, 4))
        plt.title("SVR model (online prediction)", fontsize= 15)
        plt.plot(np.arange(0, len(valid_y)), valid_y, label= "valid_y")
        plt.plot(np.arange(0, len(pred_valid)), pred_valid, label= "rolling pred_valid")
        plt.legend(loc= "best")
        plt.show()

        return model_, normalizer_x, normalizer_y

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

class DL_optimization_process:
    def __init__(self, x_cycle, x_residual, test_residual):
        self.x_cycle= x_cycle
        self.x_residual= x_residual
        self.test_residual= test_residual

    def time_sequence_silding_window(self, x, look_back):
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

    def DL_model_online_prediction(self, generated_cycle, X, y, model, valid_len):
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

    def process(self, seq_len: int, hidden_size: int, batch_size: int, cycle: np.array, x: np.array, valid_y: np.array):
        epoch_param= 1000

        x_train_Residual, y_train_Residual= time_sequence_silding_window(x= x, look_back= seq_len)
        x_train_Residual= np.append(cycle[seq_len: ].reshape(-1, 1), x_train_Residual, axis= 1)

        train_x, train_y, valid_y= x_train_Residual.reshape(-1, seq_len+ 1, 1), y_train_Residual, valid_y
        train_x, train_y, valid_y= torch.FloatTensor(train_x), torch.FloatTensor(train_y), torch.FloatTensor(valid_y)

        model_= BI_LSTM_ATTENTION(hidden_size= [1, int(hidden_size)], droprate= 0.01, num_layers= 2, device= device).to(device)

        loss_function= torch.nn.MSELoss()
        optimizer= torch.optim.Adam(model_.parameters(), lr= 0.01)

        TensorLoader= Data.TensorDataset(train_x, train_y)
        TensorLoader= Data.DataLoader(TensorLoader, batch_size= int(batch_size), shuffle= True)

        epoch_loss= []
        for i_epoch in range(1, epoch_param+ 1):
            model_.train()

            batch_loss= []
            for i, data in enumerate(TensorLoader):
                in_data, out_data= data
                in_data, out_data= in_data.to(device), out_data.to(device)
                
                optimizer.zero_grad()
                
                pred_out_data= model_(in_data)
                loss= loss_function(pred_out_data, out_data.reshape(-1, 1))

                batch_loss.append(np.sum(loss.data.cpu().numpy()))

                loss.backward()
                optimizer.step()

            model_.train()
            batch_loss= round(np.mean(batch_loss), 6)
            
            epoch_loss.append(batch_loss)

        model_.eval()
        train_x= train_x.data.cpu().numpy().reshape(-1, seq_len+ 1)

        generated_cycle = np.arange(train_x[-1, :].ravel()[0] + 0.0004, 2.0000, 0.0004)
        pred_valid= DL_model_online_prediction(generated_cycle= generated_cycle, X= train_x, y= train_y, model= model_, valid_len= len(valid_y))
        
        valid_y= valid_y.data.cpu().numpy()

        model_.eval()
        with torch.no_grad():
            train_x= torch.FloatTensor(train_x.reshape(-1, seq_len+ 1, 1))
            pred_train_y= model_(train_x.to(device)).to(device)
            pred_train_y= pred_train_y.data.cpu().numpy().ravel()

        error= round(np.abs(valid_y[-1]- pred_valid[-1]), 4)

        if error <= 0.03:
            print("Hyperparm -> seq_len_%d, hidden_size_%d, batch_size_%d"%(seq_len, hidden_size, batch_size))
            print("Bi-LSTM-AT online prediction error: ", error)
            
            dirName= os.path.join(saving_path, "residual_model_seq_len_%d_hidden_size_%d_batch_size_%d"%(seq_len, hidden_size, batch_size))

            if not os.path.exists(dirName):
                os.mkdir(dirName)
                print("Directory " , dirName ,  " Created ")
            else:    
                print("Directory " , dirName ,  " already exists")

            torch.save(model_, os.path.join(dirName, "residual_model.pth"))

            plt.figure(figsize= (13, 4))
            plt.title("Bi-LSTM-AT model (online prediction)", fontsize= 15)
            plt.plot(np.arange(0, len(valid_y)), valid_y, label= "valid_y")
            plt.plot(np.arange(0, len(pred_valid)), pred_valid, label= "rolling pred_valid")
            plt.legend(loc= "best")
            plt.savefig(os.path.join(dirName, "residual_model.png"))
            plt.show()

        elif self.iteration % 10== 0:
            # plt.figure(figsize= (13, 4))
            # plt.title("Bi-LSTM-AT model (online prediction)", fontsize= 15)
            # plt.plot(np.arange(0, len(valid_y)), valid_y, label= "valid_y")
            # plt.plot(np.arange(0, len(pred_valid)), pred_valid, label= "rolling pred_valid")
            # plt.legend(loc= "best")
            # plt.show()

            print("iteration: "+ str(self.iteration))
            print("-----------------------------------------------------------------------------------------------")
            print("Hyperparm -> seq_len: %3d, hidden_size: %3d, batch_size: %3d"%(seq_len, hidden_size, batch_size))
            print("Bi-LSTM-AT online prediction error: ", error)

        else:
            pass

        self.iteration+= 1

        return error

    def param_tuning_by_process(self, bounds):
        look_back_param= int(bounds[0])
        hidden_size_param= int(bounds[1])
        batch_size_param= int(bounds[2])
        rmse= self.process(seq_len= look_back_param, hidden_size= hidden_size_param, batch_size= batch_size_param, cycle= self.x_cycle, x= self.x_residual, valid_y=  self.test_residual)
        return rmse

    def run_optimization(self, param_bounds, max_iteration, visualization= None):
        epoch_param= 1000

        self.iteration= 1
        self.result= differential_evolution(self.param_tuning_by_process, param_bounds, maxiter= max_iteration)

        opt_look_back, opt_hidden_size, opt_batch_size= int(self.result.x[0]), int(self.result.x[1]), int(self.result.x[2])
        print("opt_look_back: %3d, opt_hidden_size: %4.6f, opt_batch_size: %4.6f"%(opt_look_back, opt_hidden_size, opt_batch_size))

        x_train_Residual, y_train_Residual= time_sequence_silding_window(x= self.x_residual, look_back= opt_look_back)
        x_train_Residual= np.append(cycle[opt_look_back: ].reshape(-1, 1), x_train_Residual, axis= 1)

        train_x, train_y, valid_y= x_train_Residual, y_train_Residual, self.test_residual

        train_x, train_y, valid_y= x_train_Residual.reshape(-1, opt_look_back+ 1, 1), y_train_Residual, valid_y
        train_x, train_y, valid_y= torch.FloatTensor(train_x), torch.FloatTensor(train_y), torch.FloatTensor(valid_y)

        model_= BI_LSTM_ATTENTION(hidden_size= [1, int(opt_hidden_size)], droprate= 0.01, num_layers= 2, device= device).to(device)

        TensorLoader= Data.TensorDataset(train_x, train_y)
        TensorLoader= Data.DataLoader(TensorLoader, batch_size= int(opt_batch_size), shuffle= True)

        loss_function= torch.nn.MSELoss()
        optimizer= torch.optim.Adam(model_.parameters(), lr= 0.01)

        epoch_loss= []
        for i_epoch in range(1, epoch_param+ 1):
            model_.train()

            batch_loss= []
            for i, data in enumerate(TensorLoader):
                in_data, out_data= data
                in_data, out_data= in_data.to(device), out_data.to(device)
                
                optimizer.zero_grad()
                
                pred_out_data= model_(in_data)
                loss= loss_function(pred_out_data, out_data.reshape(-1, 1))

                batch_loss.append(np.sum(loss.data.cpu().numpy()))

                loss.backward()
                optimizer.step()

            model_.train()
            batch_loss= round(np.mean(batch_loss), 6)
            
            epoch_loss.append(batch_loss)

            if i_epoch % 100 == 0:
                print("--> epoch: %5d // batch mse loss: %2.6f "%(i_epoch, batch_loss))

        model_.eval()

        train_x= train_x.data.cpu().numpy().reshape(-1, opt_look_back+ 1)

        generated_cycle = np.arange(train_x[-1, :].ravel()[0] + 0.0004, 2.0000, 0.0004)
        pred_valid= DL_model_online_prediction(generated_cycle= generated_cycle, X= train_x, y= train_y, model= model_, valid_len= len(valid_y))
        
        valid_y= valid_y.data.cpu().numpy()

        model_.eval()
        with torch.no_grad():
            train_x= torch.FloatTensor(train_x.reshape(-1, opt_look_back, 1))
            pred_train_y= model_(train_x.to(device)).to(device)
            pred_train_y= pred_train_y.data.cpu().numpy().ravel()

        plt.figure(figsize= (13, 4))
        plt.title("Bi-LSTM-AT model (history prediction)", fontsize= 15)
        plt.plot(np.arange(0, len(train_y)), train_y, label= "train_y")
        plt.plot(np.arange(0, len(pred_train_y)), pred_train_y, label= "pred_train_y")
        plt.legend(loc= "best")
        plt.show()

        plt.figure(figsize= (13, 4))
        plt.title("Bi-LSTM-AT model (online prediction)", fontsize= 15)
        plt.plot(np.arange(0, len(valid_y)), valid_y, label= "valid_y")
        plt.plot(np.arange(0, len(pred_valid)), pred_valid, label= "rolling pred_valid")
        plt.legend(loc= "best")
        plt.show()

        rmse= round(np.mean((valid_y- pred_valid)** 2)** (1/ 2), 4)
        print("Bi-LSTM-AT online prediction rmse: ", rmse)
        return model_

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

def DL_model_train_and_valid_processing(model, hidden_size, batch_size, epoch_param, train_x, train_y, valid_y):
    look_back_param= train_x.shape[1]
    
    train_x, train_y, valid_y= train_x.reshape(-1, look_back_param, 1), train_y, valid_y
    train_x, train_y, valid_y= torch.FloatTensor(train_x), torch.FloatTensor(train_y), torch.FloatTensor(valid_y)

    ### hidden_size= [1, hidden_size_param], droprate= dropout_param, num_layers= layer_param, device= device
    model_= model(hidden_size= [1, int(hidden_size)], droprate= 0.01, num_layers= 2, device= device).to(device)

    loss_function= torch.nn.MSELoss()
    optimizer= torch.optim.Adam(model_.parameters(), lr= 0.01)

    TensorLoader= Data.TensorDataset(train_x, train_y)
    TensorLoader= Data.DataLoader(TensorLoader, batch_size= int(batch_size), shuffle= True)

    epoch_loss= []
    for i_epoch in range(1, epoch_param+ 1):
        model_.train()

        batch_loss= []
        for i, data in enumerate(TensorLoader):
            in_data, out_data= data
            in_data, out_data= in_data.to(device), out_data.to(device)
            
            optimizer.zero_grad()
            
            pred_out_data= model_(in_data)
            loss= loss_function(pred_out_data, out_data.reshape(-1, 1))

            batch_loss.append(np.sum(loss.data.cpu().numpy()))

            loss.backward()
            optimizer.step()

        model_.train()
        batch_loss= round(np.mean(batch_loss), 6)
        
        epoch_loss.append(batch_loss)

        if i_epoch % 100 == 0:
            print("--> epoch: %5d // batch mse loss: %2.6f "%(i_epoch, batch_loss))

    model_.eval()

    train_x= train_x.data.cpu().numpy().reshape(-1, look_back_param)

    generated_cycle = np.arange(train_x[-1, :].ravel()[0] + 0.0004, 2.0000, 0.0004)
    pred_valid= DL_model_online_prediction(generated_cycle= generated_cycle, X= train_x, y= train_y, model= model_, valid_len= len(valid_y))
    
    valid_y= valid_y.data.cpu().numpy()

    model_.eval()
    with torch.no_grad():
        train_x= torch.FloatTensor(train_x.reshape(-1, look_back_param, 1))
        pred_train_y= model_(train_x.to(device)).to(device)
        pred_train_y= pred_train_y.data.cpu().numpy().ravel()

    plt.figure(figsize= (13, 4))
    plt.title("Bi-LSTM-AT model (history prediction)", fontsize= 15)
    plt.plot(np.arange(0, len(train_y)), train_y, label= "train_y")
    plt.plot(np.arange(0, len(pred_train_y)), pred_train_y, label= "pred_train_y")
    plt.legend(loc= "best")
    plt.show()

    plt.figure(figsize= (13, 4))
    plt.title("Bi-LSTM-AT model (online prediction)", fontsize= 15)
    plt.plot(np.arange(0, len(valid_y)), valid_y, label= "valid_y")
    plt.plot(np.arange(0, len(pred_valid)), pred_valid, label= "rolling pred_valid")
    plt.legend(loc= "best")
    plt.show()

    rmse= round(np.mean((valid_y- pred_valid)** 2)** (1/ 2), 4)
    print("Bi-LSTM-AT rolling prediction rmse :", rmse)

    return model_

if __name__== "__main__":
    device= "cuda" if torch.cuda.is_available() else "cpu"

    file_path= r"C:\Users\USER\Desktop\battery_sp_2\Battery_3-_4.csv"

    with open(file_path, "r") as file:
        df= pd.read_csv(file, index_col= False)

    ### SOH (state of Health)
    df_Battery_3= df[["Cycle", "#4_SOH"]].dropna()
    df_Battery_4= df[["Cycle", "#5_SOH"]].dropna()

    ############### knee-point identification ###############
    cycle= np.array(df_Battery_3["Cycle"])

    x_data= np.arange(1, len(df_Battery_3)+ 1)
    y_data= np.array(df_Battery_3["#4_SOH"])

    parameter_bound= [1, -1e-4, -1e-4, len(y_data)* 0.7]
    coe, cov= curve_fit(bacon_watts_knee_point, x_data, y_data, parameter_bound)
    opt_alpha0, opt_alpha1, opt_alpha2, opt_x1= coe

    figure_show= False

    parameter_bound= [1, -1e-4, -1e-4, len(y_data)* 0.7]
    coe, cov= curve_fit(bacon_watts_knee_point, x_data, y_data, parameter_bound)
    opt_alpha0, opt_alpha1, opt_alpha2, opt_x1= coe
    upper_x1, lower_x1= coe[3]+ 1.96* np.diag(cov)[3], coe[3]- 1.96* np.diag(cov)[3]

    ############### knee_point (bacon_watts model) ###############
    knee_point= round(opt_x1)

    bacon_watts_first_segment_curve= bacon_watts_knee_point(x_data, opt_alpha0, opt_alpha1, opt_alpha2, opt_x1)[: round(opt_x1)]
    distance_bacon_watts_degradation= ((bacon_watts_first_segment_curve- y_data[: round(opt_x1)])** 2)

    ############### knee-onset (bacon_watts model) ###############
    knee_onset= np.where(distance_bacon_watts_degradation== np.min(distance_bacon_watts_degradation[-500: ]))[0][0]

    parameter_bound= [opt_alpha0, opt_alpha1 + opt_alpha2/2, opt_alpha2, opt_alpha2/2, 0.8*opt_x1, 1.1*opt_x1]
    coe, cov= curve_fit(double_bacon_watts_model, x_data, y_data, parameter_bound)
    opt_alpha0, opt_alpha1, opt_alpha2, opt_alpha3, opt_x0, opt_x2= coe
    upper_x1, lower_x1= coe[4]+ 1.96* np.diag(cov)[4], coe[4]- 1.96* np.diag(cov)[4]

    ############### knee-onset (double bacon-watts model) ###############
    knee_onset_double_bacon_watts= round(opt_x0)

    sp= np.where(y_data<= 0.875)[0][0]

    plt.figure(num= 1, figsize= (13, 6))
    plt.plot(x_data, y_data, label= "#1_SOH")
    plt.scatter(knee_point, y_data[knee_point], label= "knee point", color= "black", s= 200, marker= "d")
    plt.scatter(knee_onset, y_data[knee_onset], label= "knee onset", color= "darkgreen", s= 200, marker= "^")
    plt.scatter(knee_onset_double_bacon_watts, y_data[knee_onset_double_bacon_watts], label= "knee onset (double Bacon Watts)", color= "darkred", s= 200, marker= "o")
    plt.scatter(sp, y_data[sp], label= "Sp", color= "red", s= 200, marker= "o")
    plt.legend(loc= "best", fontsize= 15)
    plt.xlabel("Cycle", fontsize= 15)
    plt.ylabel("SOH (state of health)", fontsize= 15)
    plt.grid(True)
    plt.show()

    print("---------------------------------------------------------------------")
    print("The optimal knee point for battery 1 locate at #", knee_point, "cycle")
    print("The optimal knee onset for battery 1 locate at #", knee_onset, "cycle")
    print("The optimal knee onset (double Bacon Watts) for battery 1 locate at #", knee_onset_double_bacon_watts, "cycle")

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
    sp= np.where(y_data<= 0.875)[0][0]

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

    ### Sliding Window For Generate Data Sequence ##################################################
    # x_train_IMFs, y_train_IMFs= time_sequence_silding_window(x= train_IMFs_1, look_back= 12)
    ### SVR Model Prediction ######################################################################
    # pre_train_ML_model, ML_normalizer_x, ML_normalizer_y= ML_model_train_and_valid_processing(gamma= 8, C= 1000, epsilon= 0.01, train_x= x_train_IMFs, train_y= y_train_IMFs, valid_y= test_IMFs_1)

    ### SVR Model Optimization ####################################################################
    ML_optimizer= ML_optimziation_process(train_IMFs_1, test_IMFs_1)
    opt_SVR_model, normalizer_x, normalizer_y= ML_optimizer.run_optimization(param_bounds= [(13, 15), (5, 15), (950, 1100), (0.008, 0.01)], max_iteration= 10000000, visualization= None)

    saving_path= r"C:\Users\USER\Desktop\battery_sp_2"
    joblib.dump(opt_SVR_model, os.path.join(saving_path, "SVR_model"))
    joblib.dump(normalizer_x, os.path.join(saving_path, "Normalizer_x_model"))
    joblib.dump(normalizer_y, os.path.join(saving_path, "Normalizer_y_model"))
    
    ### Bi-LSTM-AT Model Optimization ####################################################################
    saving_path= r"C:\Users\USER\Desktop\battery_sp_2\model_check_point_2"

    cycle= cycle* 0.0004
    cycle= cycle[Training_Cycle_Index[0]: Training_Cycle_Index[1]]
    DL_optimizer= DL_optimization_process(cycle, train_Residual, test_Residual)

    # DL_seq_len= 25; Hidden_size= 98; Batch_size= 13; Epoch_param= 100
    opt_Bi_LSTM_AT_model= DL_optimizer.run_optimization(param_bounds= [(5, 36), (100, 120), (12, 128)], max_iteration= 10000, visualization= None)

    torch.save(opt_Bi_LSTM_AT_model, os.path.join(saving_path, "residual_model.pth"))