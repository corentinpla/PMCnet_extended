# PMCnet_extended
I propose two methods of uncertainity quantification in classification tasks using BNN networks...
* The first is based on [2]. It decomposes the BNN output variation in an epistemic and in an aleatoric one.
* The second is based on [3]. It calculates uncertainty using a nearest-neighbor approach.

...as well as some improvements of PMCnet, the code implemented from this paper [1]. 
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
[1]
[2]
[3]
# Contacts
