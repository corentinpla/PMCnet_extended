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
    n_components = 100
    gmm = GaussianMixture(n_components=n_components,covariance_type="tied", random_state=42)

    # Generate sample data from the GMM
    n_samples = 300
    X = np.concatenate([np.random.normal(loc=i, scale=0.5, size=n_samples // n_components) for i in range(n_components)])

    # Fit the GMM to the data
    gmm.fit(X.reshape(-1, 1))

    # Plot the histogram of the generated data
    plt.hist(X, bins=30, density=True, alpha=0.5, color='blue', label='Histogram')

    # Plot the individual Gaussian components
    for i in range(n_components):
        mean = gmm.means_[i][0]
        std = np.sqrt(gmm.covariances_[i][0][0])
        plt.plot(np.linspace(min(X), max(X), 100), norm.pdf(np.linspace(min(X), max(X), 100), mean, std),
                color='red', linewidth=2, label=f'Component {i+1}')

    plt.title('1D Gaussian Mixture Model')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
plt.show()
