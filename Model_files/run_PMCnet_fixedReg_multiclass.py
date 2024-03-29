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
from  Model_files.functions import *
from  Model_files.PMCnet_algo import *
# from model_multiclass import *


def run_PMCnet_fixedReg_multiclass_extended(simulations,reg,loss, results_dir, x_val, y_val1,is_binary, N,K,T,sig_prop,lr,gr_period,tp,est_ml,epsilon1,epsilon2,logger):
    tp['regularization_weight'] = reg
    output_vec = []
    output_posterior_val_vec = []
    auc_mean_vec = []
    EV_auc_mean_vec=[]
    Accuracy_mean_vec = []
    EV_Accuracy_mean_vec=[]
    for i in range(simulations): 
        myprint('This is simulation {}'.format(i),logger)
        output_temp = SL_PMC_Adapt_Cov_new(N,K,T,sig_prop,lr,gr_period,tp,est_ml,epsilon1,epsilon2)
        output_vec.append(output_temp)
        # calculate the results   
        all_samples_temp = output_temp[2]
        all_weights_temp = output_temp[3]
        output_posterior_temp = []
        population = all_samples_temp[-1] # the last iteration
        weights = all_weights_temp[-1]
        output_posterior_val = BNN_posterior_multiclass_extended(N, K,population, weights, x_val, y_val1, tp)
        #print("AUC BNN",output_posterior_val[-1]["AUC"])
        #print("ACCURACY BNN",output_posterior_val[-1]["Accuracy"])
        output_posterior_val_vec.append(output_posterior_val)
        auc_mean = np.mean(output_posterior_val[3])
        Accuracy_mean = np.mean(output_posterior_val[-3])
        auc_mean_vec.append(auc_mean)
        Accuracy_mean_vec.append(Accuracy_mean)
        EV_auc_mean_vec.append(output_posterior_val[-1]["AUC"])
        EV_Accuracy_mean_vec.append(output_posterior_val[-1]["Accuracy"])
        print('The accuracy is ', EV_Accuracy_mean_vec)
        # save the results
    reg1 = np.int(np.round(reg,4)*10000)
    path_save_BNN_output  = os.path.join(results_dir,'output_glass_l2_'+str(reg1)+'.txt')             
    with open(path_save_BNN_output, "wb") as fp:   #Pickling
        pickle.dump(output_vec, fp)
    path_save_BNN_output_posterior  = os.path.join(results_dir,'output_posterior_val_glass_l2_'+str(reg1)+'.txt')             
    with open(path_save_BNN_output_posterior, "wb") as fp:   #Pickling
        pickle.dump(output_posterior_val_vec, fp)      
        
    if loss == 'AUC':
        # we maximize AUC
        
        return -np.mean(EV_auc_mean_vec)
    else:
        # we maximize Accuracy
        
        return -np.mean(EV_Accuracy_mean_vec)  
    


