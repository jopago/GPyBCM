from GPy.core.model import Model 
import GPy
import numpy as np
import multiprocessing as mp 

class FactorizedGP(Model):
    """ 
    A general model for mixtures of Gaussian Processes where
    the training is obtained via a factorized likelihood:

    p(y | x, theta) = prod_{k=1}^{M} p(y_k | x_k, theta_k) 

    here we make the usual assumptions (Liu et al. 2018) theta_k = theta for
    all k, i.e all GP have the same hyperparameters. The log-likelihood
    and its gradients are computed as the sum of the individual expert
    likelihoods 


    :param M: the number of experts 
    :param X: training set
    :param Y: training targets
    :param partition_type: how to partition data among experts
    :param kern: the covariance function of experts (if None, the kernel is RBF ARD)
    :param meanfunc: the mean function of experts

    """

    def __init__(self,X,Y, M=10, partition_type='random',
                 verbose=0, N=None,kern=None,meanfunc=None):
        super(FactorizedGP, self).__init__('FactorizedGP')


        assert X.ndim == 2
        assert Y.ndim == 2 

        if X.shape[0] != Y.shape[0]:
            print ("Size of X and y do not match!")
            return

        self.M = M
        self.partition_type = partition_type
        self.partition      = None
        self.verbose        = verbose

        self._X_shape = X.shape 
        self._Y_shape = Y.shape 

        """ Shared objects between processes """

        self.Xmap = np.memmap('_fgp_x_map',dtype='float32',mode='w+',
            shape=X.shape)
        self.Ymap = np.memmap('_fgp_y_map',dtype='float32',mode='w+',
            shape=Y.shape)

        self.Xmap[:] = X
        self.Ymap[:] = Y

        if N is None:
            self.N = int(X.shape[0] / M)
        else:
            self.N = N 

        if self.partition_type == 'random':
            self.partition  = np.random.choice(X.shape[0],
                                        size=(self.M, self.N),
                                        replace=True) 

        self.meanfunc = meanfunc

        if kern is None:
            self.kern = GPy.kern.RBF(input_dim=X.shape[1],ARD=True)
        else:
            self.kern = kern 

        self.base = GPy.models.GPRegression(X[self.partition[0]],Y[self.partition[0]],kernel=self.kern,
                    mean_function=self.meanfunc,noise_var=1e-8)
        self.base.Gaussian_noise.fix()

        self.link_parameter(self.base)

    def _log_likelihood(self,k):
        _x = self.Xmap[self.partition[k],:]
        _y = self.Ymap[self.partition[k],:]

        self.base.set_XY(_x,_y)
        
        return self.base.log_likelihood()
        
    def log_likelihood(self):
        """ 
        This function is called by GPy.core.model for optimization
        objective is negative-log likelihood of product of GPs 
        \log(\prod_{k=1}^{M} p(y_k | theta, D_k)) =
        \sum{k=1}^{n}^{M} \log p(y_k | theta, D_k) 
        we just need to sum the objective   """
        

        pool = mp.Pool() # default number of processes is number of cores

        res = pool.map(self._log_likelihood,range(self.M))
        
        pool.close()
        pool.join()
        return sum(res)

    def _log_likelihood_gradients_k(self,k):
        _x = self.Xmap[self.partition[k],:]
        _y = self.Ymap[self.partition[k],:]

        self.base.set_XY(_x,_y)

        return self.base._log_likelihood_gradients()
        
    def _log_likelihood_gradients(self):
        # called by GPy.core.Model during optimization
        pool = mp.Pool(4)

        res = pool.map(self._log_likelihood_gradients_k,range(self.M))

        pool.close()
        pool.join()

        if self.verbose >= 2:
            print(sum(res))
            
        return sum(res)
    
    def predict(self, x_new):
        # The aggregation method 

        raise NotImplementedError 
    
    
class BCM(FactorizedGP):
    """
    Bayesian Committee Machines 

    Several aggregation strategies exist, currently only the ones by
    Tresp (2000) and Deisenroth & Ng (2015) are implemented.

    :param M: the number of experts 
    :param X: training set
    :param Y: training targets
    :param partition_type: how to partition data among experts
    :param kern: the covariance function of experts (if None, the kernel is RBF ARD)
    :param meanfunc: the mean function of experts

    :param model: type of aggregation to use: 'mean' take the
    average of predictions, 'BCM' is the aggregation in Tresp (2000)
    where each gp prediction is weighted by its variance, 'rBCM' 
    is the Robust Bayesian Committee Machine of Deisenroth & Ng where
    differential entropy is used to improve weighing of experts.

    .. todo: Implement Generalized RBCM (Liu et al, 2018)
    """

    def __init__(self, X, Y, M, partition_type='random',
                 verbose=0,N=None,kern=None,meanfunc=None,
                 model='rBCM'):
    
        super(BCM,self).__init__(X,Y,M,partition_type,verbose,
             N,kern,meanfunc)
        
        self.model = model 


    def predict_k(self,k,x_new,full_cov=False):
        _x = self.Xmap[self.partition[k],:]
        _y = self.Ymap[self.partition[k],:]

        self.base.set_XY(_x,_y)

        return self.base.predict(x_new,full_cov=full_cov)
        
    def predict(self,x_new):
        assert x_new.ndim == 2

        if x_new.shape[1] != self._X_shape[1]:
            print( 'Invalid number of features')
            return 
        
        pred_mean = np.zeros(shape=(x_new.shape[0], 1))
        
        if self.model == 'mean':
            # average output of each GP

            for k in range(self.M):
                pred_mean += (1./self.M) * self.predict_k(k,x_new)[0]
                
                if self.verbose>0:
                    print ('(mean) Computing prediction')
                    
            return pred_mean
        
        if self.model == 'BCM' or self.model == 'PoE':
            """ Tresp, V. (2000). A Bayesian committee machine,
            Neural computation, 12(11), 2719-2741 """

            prior_var = np.diag(self.base.kern.K(x_new)).reshape(-1,1)+self.base.likelihood.variance
            C = (1-self.M)/prior_var.reshape(-1,1)
             
            for k in range(self.M):
                
                _pred = self.predict_k(k,x_new)
            
                C += 1./_pred[1]
            
                pred_mean += _pred[0]/_pred[1]
            
                if self.verbose > 0:
                    print ('(BCM) Computing prediction ' + str(k))
                    
            return pred_mean / C
        
        if self.model == 'rBCM':
            """ Deisenroth, M. P., & Ng, J. W. (2015). Distributed gaussian processes, 
            arXiv:1502.02843. """

            prior_var = np.diag(self.base.kern.K(x_new)).reshape(-1,1)+self.base.likelihood.variance
            C = prior_var.reshape(-1,1)
            
            for k in range(self.M):
                _pred = self.predict_k(k,x_new)
                
                # beta_k = differential entropy of predictive distribution w.r.t prior 

                beta_k = 0.5*(np.log(prior_var)-np.log(_pred[1])).reshape(-1,1) 
                
                C += beta_k*1./_pred[1] - beta_k/prior_var 
                pred_mean += (beta_k/_pred[1]) * _pred[0]
                
                if self.verbose > 0:
                    print( '(rBCM) Computing prediction ' + str(k))
                
            return pred_mean / C

    def to_dict(self):
        dict_base = self.base.to_dict()
        
        return {'Model' : self.model,
                'Base' : dict_base}