import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import numpy as np
import math
import sys
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Optimizer
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from sklearn import metrics
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy.io as sio
import time
import pandas as pd
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import pickle
from Model_files.functions import *
from Model_files.PMCnet_algo import *
from Model_files.golden_search import *
from Model_files.run_PMCnet_fixedReg_multiclass import *
import traceback
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

cuda2 = torch.device('cuda:0')

folder_name_train = '/workspace/code/PMCnet_extended/data/Glass/train_data.mat'
folder_name_val = '/workspace/code/PMCnet_extended/data/Glass/val_data.mat'
folder_name_test = '/workspace/code/PMCnet_extended/data/test_data.mat'

y_train = OpenMat(sio.loadmat(folder_name_train)['y_train'])
x_train = OpenMat(sio.loadmat(folder_name_train)['x_train'])
y_val = OpenMat(sio.loadmat(folder_name_val)['y_val'])
x_val = OpenMat(sio.loadmat(folder_name_val)['x_val'])
y_test = OpenMat(sio.loadmat(folder_name_test)['y_test'])
x_test = OpenMat(sio.loadmat(folder_name_test)['x_test'])



#scale = MinMaxScaler(feature_range = (0,1))
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_val = scale.fit_transform(x_val)
x_test = scale.fit_transform(x_test)

x_train = torch.Tensor(x_train) # transform to torch tensor
y_train = torch.Tensor(y_train-1).long().squeeze(-1)
x_val = torch.Tensor(x_val) # transform to torch tensor
y_val = torch.Tensor(y_val-1).long().squeeze(-1)
x_test = torch.Tensor(x_test) # transform to torch tensor
y_test = torch.Tensor(y_test-1).long().squeeze(-1)

print('The size of x_train is ', x_train.shape)
print('The size of y_train is ', y_train.shape)
print('The size of x_val is ', x_val.shape)
print('The size of y_val is ', y_val.shape)
print('The size of x_test is ', x_test.shape)
print('The size of y_test is ', y_test.shape)

batch_size = 10

train_dataset = TensorDataset(x_train,y_train) # create your datset
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True) 


val_dataset = TensorDataset(x_val,y_val) # create your datset
val_loader = DataLoader(val_dataset, batch_size = x_val.shape[0], shuffle = True) # not use minibatch


test_dataset = TensorDataset(x_test,y_test) # create your datset
test_loader = DataLoader(test_dataset, batch_size =  x_test.shape[0], shuffle = False) # not use minibatch

# create a dictionary to save options
hidden_layer = 10 # the number of neurons in the hidden layer # L=2

tp = {}
tp['L'] = 2
tp['S'] = 1
if tp['L'] != 1:
    tp['M'] = [x_train.shape[1],hidden_layer,6] # the map
else:
    tp['M'] = [x_train.shape[1],1] # the map
tp['prior'] = 'L2' #'no_prior', 'Gaussian_prior', 'Laplace_prior','L2'
tp['regularization_weight'] = 0.875396264914119
tp['x_0'] = x_train
tp['y'] = y_train
if tp['prior'] == 'no_prior':# MLE
    prior_W = 'no_prior'
    prior_b = 'no_prior'
    tp['prior_W'] = prior_W
    tp['prior_b'] = prior_b
elif tp['prior'] == 'Gaussian_prior':
    prior_W = isotropic_gauss_prior(mu=0, sigma=2)
    prior_b = isotropic_gauss_prior(mu=0, sigma=2)
    tp['prior_W'] = prior_W
    tp['prior_b'] = prior_b
elif tp['prior'] == 'Laplace_prior':# MAP+L1 regularization
    prior_sig = 0.1
    prior = laplace_prior(mu=0, b=prior_sig)
elif tp['prior'] == 'L2': # L2 regularization
    prior_W = 'L2'
    prior_b = 'L2'
    tp['prior_W'] = prior_W
    tp['prior_b'] = prior_b
elif tp['prior'] == 'L1': # L1 regularization
    prior_W = 'L1'
    prior_b = 'L1'
    tp['prior_W'] = prior_W
    tp['prior_b'] = prior_b       
print('The prior is ',tp['prior'])        
L = tp['L']
M = tp['M']
total_number = 0
for ll in range(L):
    total_number += M[ll]*M[ll+1]+M[ll+1]
tp['dimension'] = total_number  
tp['activation'] = ['ReLU','None']
tp['classification'] = 'multiclass'
dogolden_search = 1
dosave = 0

# some settings
use_cuda = torch.cuda.is_available()
lr = 1e-3
results_dir = 'results/Glass'  
os.makedirs(results_dir, exist_ok=True) 

# initialize the parameters by the learnt model derived by MLE
# load the parameters 

folder_name_params = '/workspace/code/PMCnet_extended/params/Glass'

W1 = OpenMat(sio.loadmat(os.path.join(folder_name_params,'W1.mat'))['W1']).cuda()
b1 = OpenMat(sio.loadmat(os.path.join(folder_name_params,'b1.mat'))['b1']).cuda()
W2 = OpenMat(sio.loadmat(os.path.join(folder_name_params,'W2.mat'))['W2']).cuda()
b2 = OpenMat(sio.loadmat(os.path.join(folder_name_params,'b2.mat'))['b2']).cuda()
  
    
print('the shape of W1 is ',W1.shape)
print('the shape of b1 is ',b1.shape)
if hidden_layer != 0:
    print('the shape of W2 is ',W2.shape)
    print('the shape of b2 is ',b2.shape)


#parameters for our algorithm
p=time.time()
N = 10 # number of proposals
K = 10  # samples per proposal per iteration
sig_prop = 0.1
lr = 2  #glocal resampling
gr_period=5
tp['regularization_weight'] = 0.9787141758189921
epsilon1 = 1e-299
epsilon2 = 1e-299

W_mu1 = W1
b_mu1 = b1

if hidden_layer != 0:
    W_mu2 = W2
    b_mu2 = b2

# est_ml is the set of all parameters (W,b), stacked in a column #16*1
if hidden_layer == 0:
    est_ml = torch.cat((torch.transpose(W_mu1,0,1).reshape(M[0]*M[1],1),b_mu1.reshape(M[1],1)),0)
else:
    est_ml1 = torch.cat((torch.transpose(W_mu1,0,1).reshape(M[0]*M[1],1),b_mu1.reshape(M[1],1)),0)
    est_ml2 = torch.cat((torch.transpose(W_mu2,0,1).reshape(M[1]*M[2],1),b_mu2.reshape(M[2],1)),0)
    est_ml = torch.cat((est_ml1,est_ml2),0)


logger = get_logger('log_BNN_multiclass_glass_l2.txt') 

if dogolden_search == 0:    
    T = 100
    N_resampled = 100
    is_binary = 0
    loss = 'Accuracy'
    y_train1 = y_train.detach().numpy()
    y_val1 = y_val.detach().numpy()
    y_test1 = y_test.detach().numpy()


    myprint('T is {}'.format(T),logger)
    myprint('regularization_weight is {}'.format(tp['regularization_weight']),logger)
    myprint('sig_prop is {}'.format(sig_prop),logger)
    myprint('N_resampled is {}'.format(N_resampled),logger)
    
    output_vec = []
    
        ##This line opens a log file
    with open("bug_log_BNN_multiclass.txt", "w") as log:

        try:
            for i in range(1): 
                myprint('This is simulation {}'.format(i),logger)
                output = SL_PMC_Adapt_Cov_new(N,K,T,sig_prop,lr,gr_period,tp,est_ml,epsilon1,epsilon2)
                output_vec.append(output) 
                best_weights=[]
                auc_mean_vec=[]
                accuracy_mean_vec=[]
                for k in range(T):
                    
                    # calculate the results   
                    all_samples_temp = output[2]
                    all_weights_temp = output[3]

                    output_posterior_temp = []
                    population = all_samples_temp[k] # the last iteration
                    weights = all_weights_temp[k]
                    
                    output_posterior_val = BNN_posterior_multiclass_extended(N, K,population, weights, x_test, y_test1, tp)
                    auc_mean = np.mean(output_posterior_val[-1]["AUC"])
                    Accuracy_mean = np.mean(output_posterior_val[-1]["accuracy"])
                    auc_mean_vec.append(auc_mean)
                    accuracy_mean_vec.append(Accuracy_mean) 

                
                data=[]
                for k in range(1,11):
                    iter=[]
                    for plt_weights in all_weights_temp:
                        plt_weights=np.sort(plt_weights)
                        iter.append(plt_weights[-k])
                    data.append(iter)



                labels = ["weight "+str(k) for k in range (1,11)]

                plt.figure(figsize=(8, 6))

                for i, array in enumerate(data):
                    plt.plot(array, marker='o', label=labels[i])

                plt.title('Weights')
                plt.xlabel('Iteration')
                plt.ylabel('Value')
                plt.yscale('log')
                plt.legend()
                plt.grid(True)
                plt.savefig("/workspace/code/weights_multiclass_extended_lr=1_sig_prop=0.1.png")

                plt.figure(figsize=(8, 6))

                #Accuracy
                plt.plot(auc_mean_vec, marker='o', label="AUC")
                plt.plot(accuracy_mean_vec, marker='o', label="Accuracy")

                plt.title('Metrics')
                plt.xlabel('Iteration')
                plt.ylabel('Value')
                #plt.yscale('log')
                plt.legend()
                plt.grid(True)
                plt.savefig("/workspace/code/metrics_multiclass_extended_lr=1_sig_prop=0.1.png")

                path_save_BNN_output  = os.path.join(results_dir,'output_glass_l2_final.txt')             
                with open(path_save_BNN_output, "wb") as fp:   #Pickling
                    pickle.dump(output_vec, fp)
            
            print("There is no bug.", file = log)
        except Exception:
            traceback.print_exc(file=log)     

else:  
    T = 5
    N_resampled = 100
    is_binary = 0
    loss = 'Accuracy'
    simulations = 5
    rangereg = [0,3]
    precision = 5e-2
    
    y_train1 = y_train.detach().numpy()
    y_val1 = y_val.detach().numpy()
    y_test1 = y_test.detach().numpy()
    
    myprint('T is {}'.format(T),logger)
    myprint('sig_prop is {}'.format(sig_prop),logger)
    myprint('N_resampled is {}'.format(N_resampled),logger)
    myprint('loss is {}'.format(loss),logger)
    myprint('Simulations is {}'.format(simulations),logger)
    myprint('the range of regularization is {}'.format(rangereg),logger)
    myprint('the precision of golden search is {}'.format(precision),logger)

    ##This line opens a log file
    with open("bug_log_BNN_multiclass.txt", "w") as log:

        try:
            crit = lambda reg: run_PMCnet_fixedReg_multiclass_extended(simulations,reg,loss, results_dir, x_test, y_test1,is_binary,N,K,T,sig_prop,lr,gr_period,tp,est_ml,epsilon1,epsilon2,logger)
            reg_final,reg_list,Loss_list = golden_search(crit,rangereg[0],rangereg[1],precision,logger)

            reg1 = np.int(np.round(reg_final,4)*10000)
            if dosave == 1:
                path_save_BNN_output  = os.path.join(results_dir,'final_output_reg_glass_l2_'+str(reg1)+'.txt')             
                with open(path_save_BNN_output, "wb") as fp:   #Pickling
                    pickle.dump(reg_list, fp)

                path_save_BNN_output = os.path.join(results_dir,'final_output_loss_glass_l2_'+str(reg1)+'.txt')           
                with open(path_save_BNN_output, "wb") as fp:   #Pickling
                    pickle.dump(Loss_list, fp)
            myprint('the final regularization weight is {}'.format(reg_final),logger)
            myprint('the reg list is {}'.format(reg_list),logger)
            myprint('the loss list is {}'.format(Loss_list),logger)  

            print("There is no bug.", file = log)
        except Exception:
            traceback.print_exc(file=log)        

