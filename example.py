from GPyBCM import BCM 
import numpy as np

if __name__ == '__main__':

	N = 10000
	x_train = np.random.normal(0,1,(N,5)) # 5 features 
	y_train = [ np.cos(np.linalg.norm(x)) for x in x_train]

	bcm = BCM(x_train, np.array(y_train).reshape(-1,1), M=40,N=300,verbose=2) # default model: rBCM 

	print('Optimizing hyperparameters..')
	bcm.optimize() # optimize sum of log-likelihood of experts 

	print(bcm.param_array)
	
	x_test = np.random.normal(0,1,(1000,5))
	y_test = [np.cos(np.linalg.norm(x)) for x in x_test]

	y_pred_rbcm = bcm.predict(x_test)
	bcm.model = 'gpoe' # generalized product of experts 
	y_pred_gpoe = bcm.predict(x_test)

	print('rBCM : ' + str(np.linalg.norm(y_test - y_pred_rbcm)))
	print('GPoE : ' + str(np.linalg.norm(y_test - y_pred_gpoe)))