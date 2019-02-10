from GPyBCM import BCM 
import numpy as np

if __name__ == '__main__':

	N = 10000
	x_train = np.random.normal(0,1,(N,5)) # 5 features 
	y_train = [ np.cos(np.linalg.norm(x)) for x in x_train]

	bcm = BCM(x_train, np.array(y_train).reshape(-1,1), M=20,N=200,verbose=2) # default model: rBCM 

	
	print('Optimizing hyperparameters..')
	bcm.optimize() # optimize sum of log-likelihood of experts 

	print(bcm.param_array)
	
	x_test = np.random.normal(0,1,(1000,5))
	y_test = [np.cos(np.linalg.norm(x)) for x in x_test]

	y_pred = bcm.predict(x_test)

	print(y_pred[0:20])
	print(y_test[0:20]) 