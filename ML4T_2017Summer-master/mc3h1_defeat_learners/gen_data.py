
import numpy as np
import math

# this function should return a dataset (X and Y) that will work
# better for linear regresstion than random trees
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    #X = np.mgrid[-5:5:0.5,-5:5:0.5].reshape(2,-1).T
    #Y = np.ones(len(X))*10.0
    X=np.random.normal(size = (100, 4))
    Y=  0.3 * X[:,0]
    return X, Y

def best4RT(seed=1489683273):
    np.random.seed(seed)
    #X = np.random.normal(size = (100, 2))
    X = np.random.random(size = (100, 2))
    Y = np.zeros(X.shape[0])
    Y=np.absolute(X[:,1]-0.5)
    return X,Y

if __name__=="__main__":
    print "they call me Tim."
