import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from scipy.special import expit
from scipy.optimize import minimize
from scipy.optimize import fmin_cg
from sklearn.linear_model import LogisticRegression


##### Now we need to load the matlab files. To do this we can use scipy
from scipy.io import loadmat

##### We also want to import the simplex downhill algorythm module

from scipy.optimize import minimize

data=loadmat("C:/Users/Rohit/Desktop/practice python/andrew_ng projects/ex3/ex3data1.mat")
weights=loadmat("C:/Users/Rohit/Desktop/practice python/andrew_ng projects\ex3\ex3weights.mat")


#### check keys
weights.keys()
data.keys()

####x and y ; for x add np.ones for constant
X=np.c_[np.ones((data['X'].shape[0],1)),data['X']]

y=data['y'].ravel()

y.shape

#####theta

Theta1=weights['Theta1']
Theta2=weights['Theta2']

Theta1.shape
Theta2.shape

############ sample space of 100 rows from # X copied code

def display(X):
    fig, arrax=plt.subplots(nrows=10,ncols=10, figsize=(10,10))

    for i in range(10):
        for j in range(10):
            row=np.random.randint(X.shape[0])
            tmp=X[row,1:].reshape(20,20, order='F')
            arrax[i,j].imshow(tmp,cmap='gray_r')
            plt.setp(arrax[i,j].get_xticklabels(), visible=False)
            plt.setp(arrax[i,j].get_yticklabels(), visible=False)
display(X)

##### next we take a look at the cost Function

N=X.shape[1]
m=X.shape[0]

initial_theta=np.zeros(N)


##### first we define the hypothesis formula
def h(Theta,X):
    return expit(np.dot(X,Theta))
####We can then calculate the cost function by spliting it up into three parts (NOT REGULARIZED FORMAT)

def cost(theta,X,y):
    m=X.shape[0]
    initial_theta=np.zeros(N)
    term1=np.dot((-1*y).T,np.log(h(theta,X)))
    term2=np.dot((1-y).T,np.log(1-h(theta,X)))
    return float((1./m)*(np.sum(term1-term2)))
cost(initial_theta,X,y)


#####gradient

y_1_hot = (y == 1).astype('float')
y_1_hot
result= minimize(cost,initial_theta,method='L-BFGS-B',args=(X,y_1_hot))



result['fun']


##### Regularization
m
lamda=1.
def cost_R(theta,X,y, lamda):
    m=X.shape[0]
    initial_theta=np.zeros(N)
    term1=np.dot((-1*y).T,np.log(h(theta,X)))
    term2=np.dot((1-y).T,np.log(1-h(theta,X)))
    term3=(lamda/2.*m)*(np.sum(theta[1:]**2))
    return float((1./m)*(np.sum(term1-term2)+term3))
cost_R(initial_theta,X,y,lamda)

def o_theta(theta,X,y,lamda):
    res= minimize(cost_R,theta,args=(X,y_1_hot,lamda),method='L-BFGS-B')
    return res.x, res.fun
theta_R, mincost_R=o_theta(initial_theta,X,y_1_hot,lamda)
mincost_R

################### One vs all Classification in order to use scipys fmin_cg which is an alternative to octaves we need to compute the gradient function

def gradient(theta,X,y,lamda=0.):
    inside=h(theta,X)-y.T
    outside=theta[1:]*(lamda/m)
    grad=((1./m)*np.dot(X.T,inside))
    grad[1:]=grad[1:]+outside
    return grad


def optimizetheta(theta,X, y, lamda=0.):
    res=fmin_cg(cost_R,fprime=gradient,x0=theta,args=(X,y,lamda),maxiter=50,disp=False,full_output=True)
    return res[0],res[1]

########This code is copied as it runs faster than mine
def buildTheta():
    """
    Function that determines an optimized theta for each class
    and returns a Theta function where each row corresponds
    to the learned logistic regression params for one class
    """
    mylambda = 0.
    initial_theta = np.zeros((X.shape[1],1)).reshape(-1)
    Theta = np.zeros((10,X.shape[1]))
    for i in xrange(10):
        iclass = i if i else 10 #class "10" corresponds to handwritten zero
        print "Optimizing for handwritten number %d..."%i
        logic_Y = np.array([1 if x == iclass else 0 for x in y])#.reshape((X.shape[0],1))
        itheta, imincost = optimizetheta(initial_theta,X,logic_Y,mylambda)
        Theta[i,:] = itheta
    print "Done!"
    return Theta
Theta = buildTheta()


#############prediction must compute hypothesis
def predict(theta,row):
    cl=[10]+range(1,10)
    hyp=np.zeros(len(cl))
    for i in range(len(cl)):
        hyp[i]=h(theta[i],row)
    return cl[np.argmax(hyp)]

################### Now we see the prediction accuracy

correct,total=0.,0.

for i in range(X.shape[0]):
    total+=1
    if predict(Theta,X[i])==y[i]:
        correct+=1
print "The test accuracy is: %0.2f%% " %(100*(correct/total))


###Now lets try using scikit learns built in logistic regression functionality and compare
clf=LogisticRegression(C=1.0,penalty='l2',multi_class='ovr')
clf.fit(X[:,1:],y)
print "Training accuracy of clf fit: %0.2f%%" % (100*clf.score(X[:,1:],y))


############### Neural Networks

thetas=[Theta1,Theta2]

def propogatef(row,theta):
    features=row
    for i in range(len(theta)):
        Theta=theta[i]
        z=np.dot(Theta,features)
        activ=expit(z)
        if i==len(theta)-1:
            return activ
        ### we need to add a bias unit
        activ=np.insert(activ,0,1)
        features=activ

def predict(row,theta):
    cl=range(1,11)
    res=propogatef(row,theta)
    return cl[np.argmax(res)]

##########test accuracy should be around 97.5%
correct, total=0.,0.

for i in range(X.shape[0]):
    total+=1
    if predict(X[i],thetas)==int(y[i]):
        correct+=1
print "The test accuracy is %0.2f%% "%(100*(correct/total))
