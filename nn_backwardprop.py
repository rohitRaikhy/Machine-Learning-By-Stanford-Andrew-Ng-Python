import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.special import expit
import random
from scipy import optimize


data=loadmat("/Users/rohitraikhy/Downloads/machine-learning-ex4/ex4/ex4data1.mat")
weights=loadmat("/Users/rohitraikhy/Downloads/machine-learning-ex4/ex4/ex4weights.mat")


weights.keys()
data.keys()
##set x (for x we need to add in ones in order to use the intercept as features) and y

data['X'].shape[0],1
X=np.c_[np.ones((data['X'].shape[0],1)),data['X']]

y=data['y'].ravel()


#### We should also now set the theta values for later use

##input layer weights
theta1=weights['Theta1']
### hidden network weights
theta2=weights['Theta2']

####Lets set up the code for dispaying the data -> This code was copied and altered from my last code

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

#### The next section requires us to compute the cost function [UNREGULARIZED]


def cost(features,classes):

    m=X.shape[0]
    #####set y values to dummy variables
    ynew=pd.get_dummies(y.ravel()).as_matrix()

    a1=features

    z2=np.dot(theta1,a1.T)
    ### add a bias unit in layer 2 and activate layer

    a2=np.c_[np.ones((features.shape[0],1)),expit(z2.T)]

    z3=np.dot(theta2,a2.T)
    a3=expit(z3)

    cost_f= (1./m)*np.sum((np.log(a3.T)*(-1*ynew)-np.log(1-a3).T*(1-ynew)))

    return cost_f
cost(X,y)

#### Now lets include regularization in our cost function


### For the backprop to work we must first introduce the sigmoid gradient function

def sigmoidgrad(z):
    return (expit(z)*(1-expit(z)))

###test to see if we're getting the correct results
[sigmoidgrad(z) for z in [-1, 0.5, 0, 0.5, 1]]

##### Now lets implement random initialition of theta values for later on when utilizing scipys optimize function

def randomtheta():
    epsilon_init = 0.12
    rand_thetas1=np.random.rand(*theta1.shape)*2*epsilon_init-epsilon_init
    rand_thetas2=np.random.rand(*theta2.shape)*2*epsilon_init-epsilon_init
    return np.concatenate((rand_thetas1.flatten(),rand_thetas2.flatten()))
randomthetas=randomtheta()


#####

t1shape=theta1.shape
t2shape=theta2.shape
###unroll the theta values

eps=0.0001
Theta1=theta1.flatten()
Theta2=theta2.flatten()
thetas=np.concatenate((Theta1,Theta2))
L=len(thetas)
thetastest=thetas.reshape((L,1))



def cost_Rtest(thetaunroll,features,y,lamda=0.):


    theta1=thetas[:10025].reshape(t1shape)
    theta2=thetas[10025:].reshape(t2shape)

    m=X.shape[0]

    ynew=pd.get_dummies(y.ravel()).as_matrix()

    a1=features

    z2=np.dot(theta1,a1.T)

    a2=np.c_[np.ones((features.shape[0],1)),expit(z2.T)]

    z3=np.dot(theta2,a2.T)
    a3=expit(z3)

    first=(1/m)*np.sum((np.log(a3.T)*(-1*ynew)-np.log(1-a3).T*(1-ynew)))
    second=(lamda/(2*m))*(np.sum((theta1**2))+np.sum((theta2**2)))
    J=first+second

    #### Now we move onto backprop -> gradient computation for unregularized

    d3=a3.T-ynew
    d2 = theta2[:,1:].T.dot(d3.T)*sigmoidgrad(z2)

    delta1=np.dot(d2,a1)
    delta2=np.dot(d3.T,a2)

    theta1grad=(delta1/m)+(theta1*lamda)
    theta2grad=(delta2/m)+(theta2*lamda)

    return J,theta1grad,theta2grad
J,grad1,grad2=cost_Rtest(thetas,X,y)
backpropgrad=np.concatenate((grad1.flatten(),grad2.flatten()))



############For just cost



def cost_Rtest1(thetaunroll,features,y,lamda):


    theta1=thetaunroll[:10025].reshape(t1shape)
    theta2=thetaunroll[10025:].reshape(t2shape)

    m=X.shape[0]

    ynew=pd.get_dummies(y.ravel()).as_matrix()

    a1=features

    z2=np.dot(theta1,a1.T)

    a2=np.c_[np.ones((features.shape[0],1)),expit(z2.T)]

    z3=np.dot(theta2,a2.T)
    a3=expit(z3)

    first=(1/m)*np.sum((np.log(a3.T)*(-1*ynew)-np.log(1-a3).T*(1-ynew)))
    second=(lamda/(2*m))*(np.sum((theta1**2))+np.sum((theta2**2)))
    J1=first+second

    return J1
cost_Rtest1(thetas,X,y,lamda=1.)


############# test to get code to work
"""
epsvec=np.zeros((L,1))
epsvec[9949]=eps
epsvec[9949]
thetastest[9949]
cost_high = cost_Rtest1(thetastest + epsvec,X,y)
cost_



epsvec=np.zeros((L,1))
epsvec[9949]=eps
newt=thetastest+epsvec
newtm=thetastest-epsvec
newt[9949]
newtm[9949]
cost_h=cost_Rtest1(newt,X,y,lamda=0)
cost_h
cost_l=cost_Rtest1(newtm,X,y,lamda=0.)
grad=(cost_h-cost_l)/float(2*eps)
grad
"""


#############

def checkgrad():
    for i in range(10):
            x = int(np.random.rand()*L)
            epsvec=np.zeros((L,1))
            epsvec[x]=eps
            testh=thetastest+epsvec
            testl=thetastest-epsvec
            costhigh=cost_Rtest1(testh,X,y,lamda=0.)
            costlow=cost_Rtest1(testl,X,y,lamda=0.)
            grad=(costhigh-costlow)/float(2*eps)
            print ("Element: %d. Numerical Gradient = %f. Backprop Gradient = %f. "%(x,grad,backpropgrad[x]))
checkgrad()

####after checking the gradients numerically we see that they are the same so can proceed
#### create backprop function seperately so I can pass through fmin_cg as it takes the gradients. copied and pasted my last code for cost (lazy solution)


def backprop(thetaunroll,features,y,lamda=0.):


    theta1=thetaunroll[:10025].reshape(t1shape)
    theta2=thetaunroll[10025:].reshape(t2shape)

    m=X.shape[0]

    ynew=pd.get_dummies(y.ravel()).as_matrix()

    a1=features

    z2=np.dot(theta1,a1.T)

    a2=np.c_[np.ones((features.shape[0],1)),expit(z2.T)]

    z3=np.dot(theta2,a2.T)
    a3=expit(z3)

    first=(1/m)*np.sum((np.log(a3.T)*(-1*ynew)-np.log(1-a3).T*(1-ynew)))
    second=(lamda/(2*m))*(np.sum((theta1**2))+np.sum((theta2**2)))
    J=first+second

    #### Now we move onto backprop -> gradient computation for unregularized

    d3=a3.T-ynew
    d2 = theta2[:,1:].T.dot(d3.T)*sigmoidgrad(z2)

    delta1=np.dot(d2,a1)
    delta2=np.dot(d3.T,a2)

    theta1grad=(delta1/m)+(theta1*lamda)
    theta2grad=(delta2/m)+(theta2*lamda)

    return np.concatenate((theta1grad.flatten(),theta2grad.flatten()))
backprop(thetas,X,y)






##############

def train_NN(lamda=0.):
    res=optimize.fmin_cg(cost_Rtest1,x0=randomthetas,fprime=backprop,args=(X,y,lamda),maxiter=50,disp=True,full_output=True)
    return res[0]
learn_thetas=train_NN()



#### to compute the training accuracy I must now create a function for forward prop
###again i have to create function for prop forward so i just copied my code

def forwardprop(thetaunroll,features):


    theta1=thetaunroll[:10025].reshape(t1shape)
    theta2=thetaunroll[10025:].reshape(t2shape)

    m=X.shape[0]

    a1=features

    z2=np.dot(theta1,a1.T)

    a2=np.c_[np.ones((features.shape[0],1)),expit(z2.T)]

    z3=np.dot(theta2,a2.T)
    a3=expit(z3)

    return a3
####
l_theta1=learn_thetas[:10025].reshape(t1shape)
l_theta2=learn_thetas[10025:].reshape(t2shape)

ntheta=[l_theta1,l_theta2]

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


#####predict classes
def predict(row,theta):
    classes=range(1,11)
    res=propogatef(row,theta)
    return classes[np.argmax(res)]

###check accuracy



correct, total =0.,0.
for i in range(X.shape[0]):
    total+=1
    if predict(X[i],ntheta)==int(y[i]):
        correct+=1
print ("The test accuracy is %0.2f%% "%(100*(correct/total)))

#####accuracy gives 97.5%
