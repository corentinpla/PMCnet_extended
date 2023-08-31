import pickle
import matplotlib.pyplot as plt 
import numpy as np 

#Load results
with open ("/workspace/code/results/Ionosphere/output_Ionosphere_l2_14164.txt","rb") as output: 
    output_PMC=pickle.load(output)

with open ("/workspace/code/results/Ionosphere/output_posterior_val_Ionosphere_l2_14164.txt","rb") as output:
    output_posterior_val=pickle.load(output)


for k in range(len(output_posterior_val)):

    print("AUC",output_posterior_val[k][-1]["AUC"])
    print("Accuracy", output_posterior_val[k][-1]["accuracy"])

#output of the neural net, mean on the weights for a given iter and a given train sample
def get_plot_output(iter, train_sample,output_posterior_val):
    labels = [0, 1]
    p=float(output_posterior_val[iter][-2][train_sample])
    values=[1-p,p]

    plt.bar(labels, values)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Bar Plot')
    plt.savefig('/workspace/code/results/plot_output.png')
    plt.close()
    print(float(output_posterior_val[iter][-2][train_sample]))


#evolution of the 10 best weights over iterations 
def get_plot_10_weights(output_PMC):

    all_weights_temp = output_PMC[3] #if simulation !=0 output_PMC[simulation][3]
    best_weights=[]
    for k in range(1,11):
        iter=[]
        for plt_weights in all_weights_temp:
            plt_weights=np.sort(plt_weights)
            iter.append(plt_weights[-k])
        best_weights.append(iter)
    
    labels = ["weight "+str(k) for k in range (1,11)]

    plt.figure(figsize=(8, 6))

    for i, array in enumerate(best_weights):
        plt.plot(array, marker='o', label=labels[i])

    plt.title('10 highest weights')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig("/workspace/code/weights.png")

#evolution of all weights over iterations
def get_plot_all_weights(output_PMC):

    all_weights_temp = output_PMC[3]

    plt.figure(figsize=(8, 6))

    for i, array in enumerate(all_weights_temp):
        plt.plot(array, "x")

    plt.title('Weights')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig("/workspace/code/weights.png")

def get_metrics(output_posterior_val):

#evolution of the metrics over iterations (T)
#plot 
    auc_mean_vec=[]
    accuracy_mean_vec=[]
    
    auc_mean = np.mean(output_posterior_val[-1]["AUC"])
    Accuracy_mean = np.mean(output_posterior_val[-1]["accuracy"])
    auc_mean_vec.append(auc_mean)
    accuracy_mean_vec.append(Accuracy_mean) 


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
    plt.savefig("/workspace/code/metrics_extended_lr=1_sig_prop=0.1.png")


