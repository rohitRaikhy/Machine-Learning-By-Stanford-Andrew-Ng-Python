import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy import optimize

X,y,c=np.loadtxt("/Users/monicaraikhy/Downloads/machine-learning-ex2 3/ex2/ex2data1.txt", delimiter=',', unpack=True)


data=pd.read_csv("/Users/monicaraikhy/Downloads/machine-learning-ex2 3/ex2/ex2data1.txt", header=None, names=['exam1','exam2','class'])

x=data[['exam1','exam2','class']]

def plot_data():
    plt.figure(figsize=(10,6))
    plt.scatter(data[data['class']==1]['exam1'],data[data['class']==1]['exam2'],marker='+',label='Admitted', c='blue')
    plt.scatter(data[data['class']==0]['exam1'],data[data['class']==0]['exam2'], label='Not Admitted', c='red')

    plt.xlabel('exam1')
    plt.ylabel('exam2')
    plt.legend()
    plt.show()
plot_data()

########sigmoid function

data=np.loadtxt('/Users/monicaraikhy/Downloads/machine-learning-ex2 3/ex2/ex2data1.txt', delimiter=',')
X=np.c_[np.ones(data.shape[0]), data[:,0:2]]
X
y=np.c_[data[:,2]]


########### hypothesis
def hyp(Theta,x):
    return expit(np.dot(x,Theta))

################### cost function


m=X.shape[0]

def computecost(Theta,x,y):
    term1=np.dot((-1*y).T,np.log(hyp(Theta,x)))
    term2=np.dot((1-y).T,np.log(1-hyp(Theta,x)))
    return float((1./m)*(np.sum(term1 - term2)))

initial_theta=np.zeros((X.shape[1],1))
computecost(inital_theta,X,y)

#########gradient Descent using built in function in scipy

def optimizetheta(Theta,X,y):
    test=optimize.fmin(computecost,x0=Theta,args=(X,y), maxiter=400, full_output=True)
    return test[0],test[1]

theta, mincost=optimizetheta(initial_theta,X,y)

print computecost(theta,X,y)

#########################
