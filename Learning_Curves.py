import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import optimize
from scipy.optimize import minimize

data=loadmat("/Users/rohitraikhy/Downloads/machine-learning-ex5 3/ex5/ex5data1.mat")

trainX=np.c_[np.ones((data['X'].shape[0],1)),data['X']]
trainY=data['y']

xval=np.c_[np.ones((data['Xval'].shape[0],1)),data['Xval']]
yval=data['yval']


plt.scatter(trainX[:,1],trainY,s=30,c='r',marker='x',linewidths=1)
plt.xlim(-50,40)
plt.xlabel('change in water level (x)')
plt.ylabel('water flowing from out of the damm (y)')


##hypothesis
def hyp(theta,X):
    return np.dot(X,theta)

m=trainY.size

def costR(theta,X,y,lamda=0.):
    myh = hyp(theta,X).reshape((m,1))
    part1=float((1./(2*m))*np.dot((myh-y).T,(myh-y)))
    part2=(float(lamda)/(2*m))*float(np.dot(theta[1:].T,theta[1:]))
    return float(part1 + part2)

initialtheta=np.ones((trainX.shape[1],1))

mytheta = np.array([[1.],[1.]])
costR(mytheta,trainX,trainY,lamda=1.)



def costRtest(theta,X,y,lamda=0.):
    m=X.shape[0]
    myh = hyp(theta,X).reshape((m,1))
    part1=float((1./(2*m))*np.dot((myh-y).T,(myh-y)))
    part2=(float(lamda)/(2*m))*float(np.dot(theta[1:].T,theta[1:]))
    return float(part1 + part2)

initialtheta=np.ones((trainX.shape[1],1))

mytheta = np.array([[1.],[1.]])
costRtest(mytheta,trainX,trainY,lamda=1.)

####Now we move onto calculating the gradient

def test(theta,X,y,lamda=0.):
    myh=hyp(theta,X).reshape((m,1))
    grad=(float((1./m))*np.dot((hyp(theta,X)-y).T,X)).T
    reg=(float(lamda/m))*theta
    reg[0]=0
    result=grad+reg
    return result.flatten()
test(initialtheta,trainX,trainY,lamda=1.)



#### Next we move onto optimizing the values of theta using scipys built in functionality fmin_cg

##used costRtest not costtest I am going to have to change this back

def optimizeT(theta,X,y,lamda=0.):
    res=minimize(costRtest,x0=theta,args=(X,y,lamda),method='L-BFGS-B')
    return res, res.x
full_output, theta =optimizeT(initialtheta,trainX,trainY,lamda=0.)

def linear_model(x_num):
    return (theta[0]+theta[1]*x_num)

plt.figure(figsize=(10,6))
plt.scatter(trainX[:,1],trainY,s=30,c='r', marker='x', linewidths=1)
plt.xlim(-50,40)
plt.grid(True)
plt.xlabel('change in water level (x)')
plt.ylabel('water flowing from out of the damm (y)')
plt.plot(trainX[:,1],linear_model(trainX[:,1]), label="linear_")


###Now we move onto the bias variance section

#subdivide the data set into a training set and cross validation set. traininh set X iterating over one each time


def plot_learning():

    newm=[]
    train_error=[]
    CV_error=[]

    for i in range(1,trainX.shape[0]+1,1):
        trainsub=trainX[:i,:]
        ysub=trainY[:i]
        newm.append(ysub.shape[0])
        outputs, fitthetas=optimizeT(initialtheta,trainsub,ysub,lamda=0.)
        train_error.append(costRtest(fitthetas,trainsub,ysub,lamda=0.))
        CV_error.append(costRtest(fitthetas,xval,yval,lamda=0.))

    plt.figure(figsize=(8,5))
    plt.plot(newm,train_error,label='Training error')
    plt.plot(newm,CV_error,label='Cross Validation error',color='red')
    plt.legend()
    plt.title('Learning Curves')
    plt.xlabel('number of training examples')
    plt.ylabel('Error Value')
    plt.grid(True)
    plt.ylim(0,100)
plot_learning()

### From the leaning curves it looks like the model has a high bias problem. Therefore we should look into using a different linear_model

##next we are going to map the data onto a 8th degree polynomial to see if this will give us a better linear_model
## Looking at the linear model it is clear that this was caused by a high degree of underfitting of the data

###feature mapping of degree 8


def map_poly(X,d):

    newx=X.copy()
    for i in range(d-1):
        newx=np.insert(newx,newx.shape[1],np.power(newx[:,1],i+2),axis=1)
    return newx
global_d=8

maptrainX=map_poly(trainX,global_d)
mapCVX=map_poly(xval,global_d)


### After mapping we realize we need to perform feature normaliztion

feature_trainX=maptrainX[:,1:]
feature_xval=mapCVX[:,1:]


def  featureNormalize(x):

    # compute mean and standard deviation
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)

    X_norm = (x - mu) / sigma

    return X_norm, mu, sigma
feature_Xtrain, mu, sigma = featureNormalize(feature_trainX)
feature_Xval, mu2, sigma2 =featureNormalize(feature_xval)


feature_Xtrain=np.hstack(((np.ones((trainX.shape[0],1)), feature_Xtrain)))
feature_Xval=np.hstack(((np.ones((xval.shape[0],1)), feature_Xval)))


####Now we check the learning curves without regulalriztion must add more terms to inital theta values as more features were added


def plot_learning():


    newm=[]
    train_error=[]
    CV_error=[]

    initial_thetas=np.ones((global_d+1,1))
    for i in range(1,feature_Xtrain.shape[0]+1,1):
            train_subset=feature_Xtrain[:i,:]
            ysub=trainY[:i]
            newm.append(ysub.shape[0])
            outputs, fitthetas=optimizeT(initial_thetas,train_subset,ysub,lamda=0.)
            train_error.append(costRtest(fitthetas,train_subset,ysub,lamda=0.))
            CV_error.append(costRtest(fitthetas,feature_Xval,yval,lamda=0.))

    plt.figure(figsize=(8,5))
    plt.plot(newm,train_error,label='Training error')
    plt.plot(newm,CV_error,label='Cross Validation error',color='red')
    plt.legend()
    plt.title('Learning Curves')
    plt.xlabel('number of training examples')
    plt.ylabel('Error Value')
    plt.grid(True)

plot_learning()

###Copied code from above just to get the thetas
mytheta=np.ones((feature_Xtrain.shape[1],1))
output,fit_theta=optimizeT(mytheta,feature_Xtrain,trainY,1.)
fit_theta

feature_Xtrain


def plotg():
    plt.figure(figsize=(10,6))
    plt.scatter(maptrainX[:,1],trainY,s=30,c='r', marker='x', linewidths=1)
    plt.grid(True)
    plt.xlabel('change in water level (x)')
    plt.ylabel('water flowing from out of the damm (y)')
plotg()


def plotFit():

    n_points_to_plot = 50
    xvals = np.linspace(-55,55,n_points_to_plot)

    xmat=np.ones((n_points_to_plot,1))

    xmat = np.insert(xmat,xmat.shape[1],xvals.T,axis=1)

    xmat = map_poly(xmat,8)
    xmat=xmat[:,1:]



    feature_Xtest, mutest, sigmatest=featureNormalize(xmat)
    feature_Xtest=np.hstack(((np.ones((xmat.shape[0],1)), feature_Xtest)))

    plotg()
    plt.plot(xvals,hyp(fit_theta,feature_Xtest),'b--')

plotFit()



def plot_learning():


    newm=[]
    train_error=[]
    CV_error=[]

    initial_thetas=np.ones((global_d+1,1))
    for i in range(1,feature_Xtrain.shape[0]+1,1):
            trainsub=feature_Xtrain[:i,:]
            ysub=trainY[:i]
            newm.append(ysub.shape[0])
            outputs, fitthetas=optimizeT(initial_thetas,trainsub,ysub,lamda=1.)
            train_error.append(costRtest(fitthetas,trainsub,ysub,lamda=1.))
            CV_error.append(costRtest(fitthetas,feature_Xval,yval,lamda=1.))

    plt.figure(figsize=(8,5))
    plt.plot(newm,train_error,label='Training error')
    plt.plot(newm,CV_error,label='Cross Validation error',color='red')
    plt.legend()
    plt.title('Learning Curves')
    plt.xlabel('number of training examples')
    plt.ylabel('Error Value')
    plt.grid(True)

plot_learning()
