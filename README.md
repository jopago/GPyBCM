# Bayesian Committee Machines

Gaussian Process regression scales poorly with dimension as it requires to invert an NxN symmetric matrix, where N is the size of the training set. Bayesian Committee Machines (BCM) are ensemble methods that aggregate individual GP predictions (Tresp, 2000) and allow distributed training (Deisenroth et al., 2015) and prediction. 

A BCM is made of M "experts" sharing the same kernel and hyperparemeters. The likelihood of a BCM is factorized over each expert to yield:

![Alt text](https://latex.codecogs.com/gif.latex?p%28y%20%5Cmid%20X%2C%5Ctheta%29%20%3D%20%5Cprod_%7Bi%3D1%7D%20p%5Cleft%28y_i%20%5Cmid%20X_i%2C%20%5Ctheta%29)

Thus the log-likelihood is just a sum of individual likelihood and the same goes for its gradients, meaning that during training, i.e finding the best hyperparameters, gradients can be computed in parallel. 


# References 

[1]: Tresp, V, A Bayesian Committee Machine, Neural Computation. http://www.dbs.ifi.lmu.de/~tresp/papers/bcm6.pdf

[2] Deisenroth, M. P., & Ng, J. W. (2015). Distributed gaussian processes. arXiv:1502.02843.
            
