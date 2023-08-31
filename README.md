# PMCnet_extended
### I propose two methods of uncertainity quantification in classification tasks using BNN networks...
* The first is based on [2]. It decomposes the BNN output variation in an epistemic and in an aleatoric one.
* The second is based on [3]. It calculates uncertainty using adaptative neighborhood aggregation.

### ...as well as some improvements of PMCnet, the code implemented from this paper [1]. 
* A new approach for metrics calculations, see `BNN_posterior_extended()` 
* Bug fixes for sampling importance points
* A second variance reduction method : simulated annealing
* Add options for resampling
  * local `lr=1`
  * global `lr=0`
  * both `lr=2`
  * no resampling `lr=-1`
* A results visualisation module, see `results.py`



# Bibliography 
* [1][Efficient Bayes Inference in Neural Networks through Adaptive Importance Sampling](https://arxiv.org/pdf/2210.00993.pdf), Y. Huang , E. Chouzenoux, V. Elvira, J-C. Pesquet 
* [2][Uncertainty quantification using Bayesian neural networks in classification: Application to biomedical image segmentation](https://www.sciencedirect.com/science/article/abs/pii/S016794731930163X?casa_token=VXK819Q9recAAAAA:osoHbK0r-7Jmue5ta272UZ41pb2HfPUaa7rfmJp2eSRV2W-q4NzqojzwVxz63ISeSsoB5vmzCg8) 
* [3][Birds of a Feather Trust Together: Knowing When to Trust a Classifier via Adaptive Neighborhood Aggregation](https://arxiv.org/pdf/2211.16466.pdf)
# Contacts

