import pickle
import matplotlib.pyplot as plt 

with open ("/workspace/code/results/Glass/output_glass_l2_20213.txt","rb") as output: 
    output_PMC=pickle.load(output)

with open ("/workspace/code/results/Glass/output_posterior_val_glass_l2_20213.txt","rb") as output:
    output_posterior_val=pickle.load(output)


for k in range(len(output_posterior_val)):

    print("AUC",output_posterior_val[k][-1]["AUC"])
    print("Accuracy", output_posterior_val[k][-1]["Accuracy"])

#print output of the neural net, moyennée sur les weights

def get_plot_multiclass(iter, train_sample):
    labels = [1,2,3,4,5,6]
    proba=output_posterior_val[iter][-2][train_sample].tolist()

# Create the bar plot
    plt.bar(labels, proba)
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Bar Plot')
    plt.savefig('/workspace/code/results/plot.png')
    plt.close()
