import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import pickle 

with open ("/workspace/code/results/autoMPG/output_autoMPG_l2_final.txt","rb") as output: 
    output_PMC=pickle.load(output)

with open ("/workspace/code/results/autoMPG/output_psoterior_val_autoMPG_l2_final.txt","rb") as output:
    output_posterior_val=pickle.load(output)

def get_mixture(output_posterior_val,test_sample):
    # Create a Gaussian Mixture Model with 3 components
    n_components = 10
    n_samples = 5000

    #compute the 10 bests weights
    wn=output_posterior_val[-1][1]  #last itération 
    best_indices_wn=np.argsort(wn)[::-1]
    best_indices_wn=best_indices_wn[:10]

    #output of the neural net associated with the 10 bests weights 
    output=output_posterior_val[-1][0][test_sample]
    print(len(output_posterior_val[-1]))
    print(len(output_posterior_val[-1][0]))
    print(len(output_posterior_val[-1][0][-1]))


    weights=[wn[best_indices_wn[k]] for k in range(n_components)]
    means=[output_posterior_val[-1][0][best_indices_wn[k]][test_sample] for k in range(n_components)]
    stds=[0.1]*10

    
    # Generate random samples from the Gaussian Mixture Model
    samples = []
    for i in range(n_components):
        component_samples = np.random.normal(loc=means[i].cpu().detach().numpy(), scale=stds[i], size=int(n_samples * weights[i]))
        samples.extend(component_samples)

    # Plot the histogram of the generated data
    plt.hist(samples, bins=300, density=True, alpha=0.5, color='blue', label='Histogram')

    # Plot the individual Gaussian components with weights
    for i in range(n_components):
        weight = weights[i]
        mean = means[i]
        std = stds[i]
        plt.plot(np.linspace(min(samples), max(samples), 100), weight * norm.pdf(np.linspace(min(samples), max(samples), 100), mean.cpu().detach().numpy(), std),
                color='red', linewidth=2, label=f'Component {i+1} (Weight {weight:.2f})')

    plt.title('Generated Gaussian Mixture Model')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig("/workspace/code/results/figures/mixture.png")

get_mixture(output_posterior_val,1)