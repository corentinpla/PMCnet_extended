import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import pickle 

with open ("/workspace/code/results/Glass/output_autoMPG_l2.txt","rb") as output: 
    output_PMC=pickle.load(output)

with open ("/workspace/code/results/Glass/output_posterior_val_autoMPG_l2.txt","rb") as output:
    output_posterior_val=pickle.load(output)

def get_mixture(output_posterior_val):
    # Create a Gaussian Mixture Model with 3 components
    n_components = 10
    n_samples = 5000

    #compute the 10 bests weights
    wn=output_posterior_val[-1][1]
    best_indices_wn=np.argsort(wn)[::-1]
    best_indices_wn=best_indices_wn[:10]

    #output of the neural net associated with the 10 bests weights 
    

    weights=[wn[best_indices_wn[k]] for k in range(n_components)]
    means=[]
    stds=[0.1]*10

    
    # Generate random samples from the Gaussian Mixture Model
    samples = []
    for i in range(n_components):
        component_samples = np.random.normal(loc=means[i], scale=stds[i], size=int(n_samples * weights[i]))
        samples.extend(component_samples)

    # Plot the histogram of the generated data
    plt.hist(samples, bins=30, density=True, alpha=0.5, color='blue', label='Histogram')

    # Plot the individual Gaussian components with weights
    for i in range(n_components):
        weight = weights[i]
        mean = means[i]
        std = stds[i]
        plt.plot(np.linspace(min(samples), max(samples), 100), weight * norm.pdf(np.linspace(min(samples), max(samples), 100), mean, std),
                color='red', linewidth=2, label=f'Component {i+1} (Weight {weight:.2f})')

    plt.title('Generated Gaussian Mixture Model')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
