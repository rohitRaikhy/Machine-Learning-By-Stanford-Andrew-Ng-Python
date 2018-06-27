import pandas as pd

import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)

### warm up exercise
def identity(n):
    return np.identity(n)
identity(5)

###### Linear Regression plotting the data

data=np.loadtxt('/Users/monicaraikhy/Desktop/python/machine-learning-ex1/ex1/ex1data1.txt',delimiter=',')


X = np.c_[np.ones(data.shape[0]),data[:,0]]
y = np.c_[data[:,1]]



plt.scatter(X[:,1],y,s=30,c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000')
plt.ylabel('Profit in $10,000')

#### gradient descent -- cost function with theta initialized at 0,0

iterations=1500
alpha=0.01

m=y.size


def hyp(theta,X):
    return np.dot(X,theta)

def cost_function(mytheta,X,y): #Cost function

    return float((1./(2*m)) * np.dot((hyp(mytheta,X)-y).T,(hyp(mytheta,X)-y)))

initial_theta=np.zeros((X.shape[1],1))
cost_function(initial_theta,X,y)
###### gradient descenct algorythm


def gradient_descent(X,theta_start=np.zeros(2)):
    theta_values=[]
    jvec=[]
    theta=theta_start
    for i in xrange(iterations):
        tmptheta=theta
        jvec.append(cost_function(theta,X,y))
        theta_values.append(list(theta[:,0]))
        for j in xrange(len(tmptheta)):
            tmptheta[j]=theta[j]-(alpha/m)*np.dot((hyp(initial_theta,X)-y).T,np.array(X[:,j]).reshape(m,1))
        theta=tmptheta
    return theta, jvec, theta_values
theta,jvec, theta_values =gradient_descent(X,initial_theta)





theta_values
theta
jvec
##### plotting cost function with respect to cost_iterations


def plotConver(jvec):
    plt.grid(True)
    plt.plot(jvec)
    plt.ylim(4,7)
    plt.xlim(0,1500)
    plt.xlabel("iterations")
    plt.ylabel("Cost")

plotConver(jvec)


def linear_model(x_num):
    return (theta[0]+theta[1]*x_num)


plt.figure(figsize=(10,6))
plt.scatter(X[:,1],y,s=30,c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.grid(True)
plt.xlabel('Population of City in 10,000')
plt.ylabel('Profit in $10,000')
plt.plot(X[:,1],linear_model(X[:,1]),label='linear_')



####predictions
linear_model((35000,70000))


###### plot contour graphs
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import itertools


fig=plt.figure(figsize=(12,12))
ax=fig.gca(projection='3d')

xvals=np.arange(-10,10,.5)
yvals=np.arange(-1,4,.1)

xs=[]
ys=[]
zs=[]

for i in xvals:
    for j in yvals:
        xs.append(i)
        ys.append(j)
        zs.append(cost_function(np.array([[i],[j]]),X,y))

scat = ax.scatter(xs,ys,zs,c=np.abs(zs),cmap=plt.get_cmap('YlOrRd'))

plt.xlabel('theta0',fontsize=10)
plt.ylabel('theta 1',fontsize=10)
plt.title('Gradient Descent Graph',fontsize=20)
plt.plot([x[0] for x in theta_values],[x[1] for x in theta_values],jvec,'bo-')
plt.show()



################## 2nd example of code


################



################multiple variables


datafile='/Users/monicaraikhy/Desktop/python/machine-learning-ex1/ex1/ex1data2.txt'

data = np.loadtxt(datafile, delimiter=',')

data
X, y = data[:,:-1], data[:,-1]
m = len(y)
y = np.c_[data[:,2]]


def  featureNormalize(x):

    # compute mean and standard deviation
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    X_norm = (x - mu) / sigma

    return X_norm, mu, sigma
X_test, mu, sigma = featureNormalize(X)


X_test= np.hstack(((np.ones((X.shape[0],1)), X_test)))

X_test


initial_theta=np.zeros(((X_test.shape[1],1)))

initial_theta

theta, jvec,theta_values= gradient_descent(X_test,initial_theta)
jvec



def plotconver2 (jvec):
    plt.grid(True)
    plt.xlim(0,1500)
    plt.xlabel("iterations")
    plt.ylabel("Cost")
    plt.plot(jvec)
plotconver2(jvec)


########normal equation

def nomral_eq(X,y):
    inv_term=np.linalg.inv(np.dot(X.T,X))
    theta=np.dot(inv_term,np.dot(X.T,y))
    return theta


######Un-normalize data


X=np.hstack((np.ones((X.shape[0],1)),X))


theta_norm_eq=nomral_eq(X,y)
theta_norm_eq
