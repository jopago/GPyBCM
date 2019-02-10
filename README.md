

# GPyBCM
> Bayesian Committee Machines for large-scale Gaussian Process regression with GPy and multiprocessing

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

# Example 
```python
from GPyBCM import BCM

bcm = BCM(x_train, y_train, M=50,N=250) 
# M: number of GP experts, N: size of each expert's training set 
bcm.optimize()

y_pred = bcm.predict(x_test)
```


# Bayesian Committee Machines

Gaussian Process regression scales poorly with dimension as it requires to invert an NxN symmetric matrix, where N is the size of the training set. Bayesian Committee Machines (BCM) are ensemble methods that aggregate individual GP predictions (Tresp, 2000) and allow distributed training (Deisenroth et al., 2015). 

A BCM is made of M "experts" sharing the same kernel and hyperparemeters but each having its own training set which is a subset of the global and large initial one. Usually the partitioning of data between experts is done randomly. The likelihood of a BCM is factorized over each expert to yield:

![Alt text](https://latex.codecogs.com/gif.latex?p%28y%20%5Cmid%20X%2C%5Ctheta%29%20%3D%20%5Cprod_%7Bi%3D1%7D%20p%5Cleft%28y_i%20%5Cmid%20X_i%2C%20%5Ctheta%29)

Thus the log-likelihood is just a sum of individual likelihoods and the same goes for its gradients, meaning that during training (which consists in finding the best kernel parameters) gradients can be computed in parallel. 




# References 

[1]: Tresp, V, A Bayesian Committee Machine, Neural Computation. http://www.dbs.ifi.lmu.de/~tresp/papers/bcm6.pdf

[2] Deisenroth, M. P., & Ng, J. W. (2015). Distributed gaussian processes. arXiv:1502.02843.
            
