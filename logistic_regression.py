import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
from scipy import optimize

dt="C:/Users/Rohit/Desktop/practice python/andrew_ng data/ex2/ex2data1.txt"

X,y,c=np.loadtxt(dt, delimiter=',', unpack=True)


data=pd.read_csv(dt, header=None, names=['exam1','exam2','class'])

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

data_m=np.loadtxt(dt, delimiter=',')
X=np.c_[np.ones(data_m.shape[0]), data_m[:,0:2]]

y=np.c_[data_m[:,2]]


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
computecost(initial_theta,X,y)


#########gradient Descent using built in function in scipy

def o_theta(Theta,X,y):
    test=optimize.fmin(computecost,x0=Theta,args=(X,y), maxiter=400, full_output=True)
    return test[0],test[1]

theta, mincost=o_theta(initial_theta,X,y)

################# prediction

hyp(theta,np.array([1,45.,85.]))


####################



print computecost(theta,X,y)

#########################

def plot_model():
    plt.figure(figsize=(10,6))
    plt.scatter(data[data['class']==1]['exam1'],data[data['class']==1]['exam2'],marker='+',label='Admitted', c='blue')
    plt.scatter(data[data['class']==0]['exam1'],data[data['class']==0]['exam2'], label='Not Admitted', c='red')

    plt.xlabel('exam1')
    plt.ylabel('exam2')
    plt.legend()

    x=np.linspace(30,100,500)
    y= -(theta[0]+theta[1]*x)/theta[2]
    plt.plot(x,y)
    plt.show()
plot_model()




####################### Regularization with logsitic Regression

dt2="C:/Users/Rohit/Desktop/practice python/andrew_ng data/ex2/ex2data2.txt"

data2=pd.read_csv(dt2,header=None, names=['test1','test2','class'])
data2
def plot2():
    plt.figure(figsize=(10,6))
    plt.scatter(data2[data2['class']==1]['test1'],data2[data2['class']==1]['test2'], marker='+',label='accepted',c='blue')
    plt.scatter(data2[data2['class']==0]['test1'],data2[data2['class']==0]['test2'], label='rejected', c='red')

    plt.xlabel('test1')
    plt.ylabel('test2')
    plt.legend()
    plt.show()
plot2()


X1,y1,z1=np.loadtxt(dt2,delimiter=',',unpack=True)
d2=np.loadtxt(dt2,delimiter=',')

i2=np.zeros((x.shape[1]))

X1=np.c_[np.ones(d2.shape[0]),d2[:,0:2]]

y1=np.c_[d2[:,2]]

############feature mapping copied code

from numpy import ones, append, array


def map_feature(x1, x2):
    '''
    Maps the two input features to quadratic features.
    Returns a new feature array with more features, comprising of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
    Inputs X1, X2 must be the same size
    '''
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    out = ones(shape=(x1[:, 0].size, 1))

    m, n = out.shape

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = append(out, r, axis=1)

    return out
mapped=map_feature(X1[:,1],X1[:,2])

mapped

##################cost
inital_theta=np.zeros((mapped.shape[1],1))

def cost_R(Theta,x,y, lamda=0.):
    m=X1.shape[0]
    term1=np.dot((-1*y).T,np.log(hyp(Theta,x)))
    term2=np.dot((1-y).T,np.log(1-hyp(Theta,x)))
    term3=(lamda/2)*np.sum(np.dot(Theta[1:].T,Theta[1:]))
    return float((1./m)*(np.sum(term1-term2)+term3))
cost_R(inital_theta,mapped,y1)

#########################

def o_theta1(mytheta, myX, myy, mylambda=0.):
    test1=optimize.minimize(cost_R, mytheta,args=(myX,myy, mylambda),options={"maxiter":700})
    return test1.x, test1.fun

theta_R, mincost_R=o_theta1(inital_theta,mapped,y1)
theta_R


#################### Part of this code is copied


def plotBoundary(mytheta, myX, myy, mylambda=0.):
    plt.figure(figsize=(10,6))
    plt.scatter(data2[data2['class']==1]['test1'],data2[data2['class']==1]['test2'], marker='+',label='accepted',c='blue')
    plt.scatter(data2[data2['class']==0]['test1'],data2[data2['class']==0]['test2'], label='rejected', c='red')

    plt.xlabel('test1')
    plt.ylabel('test2')
    plt.legend()
    """
    Function to plot the decision boundary for arbitrary theta, X, y, lambda value
    Inside of this function is feature mapping, and the minimization routine.
    It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
    And for each, computing whether the hypothesis classifies that point as
    True or False. Then, a contour is drawn with a built-in pyplot function.
    """
    theta_R, mincost_R = o_theta1(mytheta,myX,myy,mylambda)
    xvals = np.linspace(-1,1.5,50)
    yvals = np.linspace(-1,1.5,50)
    zvals = np.zeros((len(xvals),len(yvals)))
    for i in xrange(len(xvals)):
        for j in xrange(len(yvals)):
            myfeaturesij = map_feature(np.array([xvals[i]]),np.array([yvals[j]]))
            zvals[i][j] = np.dot(theta_R,myfeaturesij.T)
    zvals = zvals.transpose()

    u, v = np.meshgrid( xvals, yvals )
    mycontour = plt.contour( xvals, yvals, zvals,[0])
    #Kind of a hacky way to display a text on top of the decision boundary
    myfmt = { 0:'Lambda = %d'%mylambda}
    plt.clabel(mycontour, inline=1, fontsize=15, fmt=myfmt)
    plt.title("Decision Boundary")


plt.figure(figsize=(12,10))
plotBoundary(theta_R,mapped,y1,0.)
