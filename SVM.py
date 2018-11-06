import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import svm

d1="/Users/rohitraikhy/Downloads/machine-learning-ex6 3/ex6/ex6data1.mat"
data1=loadmat(d1)
data1

X=data1["X"]
Y=data1["y"]

###Seperate data between classes 1 and 0

class1=np.array([X[i] for i in range(len(X)) if Y[i]==1 ])
class2=np.array([X[i] for i in range(len(X)) if Y[i]==0])

class1X1=class1[:,0]
class1X2=class1[:,1]

class2X1=class2[:,0]
class2X2=class2[:,1]

def plot():
    plt.figure(figsize=(10,6))
    plt.scatter(class1X1,class1X2,s=30,c='b',marker='x',linewidths=1)
    plt.scatter(class2X1,class2X2, s=30,c='r',marker='o',linewidths=1)
plot()
##We need to make the function to plot the decision boundary

def plot_svm(svm):
    xvals = np.linspace(0,4.5,100)
    yvals = np.linspace(1.5,5,100)
    zvals = np.zeros((len(xvals),len(yvals)))
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            zvals[i][j] = float(svm.predict(np.array([[xvals[i],yvals[j]]])))
    zvals = zvals.transpose()
    contour=plt.contour(xvals,yvals,zvals,[0])
    plt.title('Decision Boundary')




####Now we train the data to fit a linear model

linear_SVM=svm.SVC(C=1,kernel='linear')
linear_SVM.fit(X,Y.flatten())

plot()
plot_svm(linear_SVM)

#### C=100 classifes all data correctly but appears not to be a natural fit

linear_svm2=svm.SVC(C=100,kernel='linear')
linear_svm2.fit(X,Y.flatten())

plot()
plot_svm(linear_svm2)

####Now we make the gausssian function

def gaussian(X1,X2,sigma):
    part1=-1*np.dot((X1-X2),(X1-X2))
    part2=2*(sigma**2)
    g=np.exp(part1/part2)
    return g
gaussian(np.array([1, 2, 1]),np.array([0, 4, -1]), 2.)

####Now we move onto dataset 2

data2=loadmat("/Users/rohitraikhy/Downloads/machine-learning-ex6 3/ex6/ex6data2.mat")
data2.keys()
##Plot the data

Xd2=data2['X']
Yd2=data2['y']

c1=np.array([Xd2[i] for i in range(len(Xd2)) if Yd2[i]==1 ])
c2=np.array([Xd2[i] for i in range(len(Xd2)) if Yd2[i]==0])

c1X1=c1[:,0]
c1X2=c1[:,1]

c2X1=c2[:,0]
c2X2=c2[:,1]

def plot2():
    plt.figure(figsize=(10,6))
    plt.scatter(c1X1,c1X2,s=30,c='b',marker='x',linewidths=1)
    plt.scatter(c2X1,c2X2, s=30,c='r',marker='o',linewidths=1)
    plt.title("DataSet2")
plot2()

##Now we use scipys built in function for gaussian kernels to train the data and plot the decision boundary

##fit the model
sigma=0.05
gamma=np.power(sigma,-2)

g_svm=svm.SVC(C=1,kernel='rbf',gamma=gamma)
g_svm.fit(Xd2,Yd2.flatten())

##Now we plot the data

def plot_svm2(svm):
    xvals = np.linspace(0,1.0,100)
    yvals = np.linspace(0.4,1.0,100)
    zvals = np.zeros((len(xvals),len(yvals)))
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            zvals[i][j] = float(svm.predict(np.array([[xvals[i],yvals[j]]])))
    zvals = zvals.transpose()
    contour=plt.contour(xvals,yvals,zvals,[0])
    plt.title('Decision Boundary')

plot2()
plot_svm2(g_svm)


#### Next we load the data for the third data set

data3=loadmat("/Users/rohitraikhy/Downloads/machine-learning-ex6 3/ex6/ex6data3.mat")

Xd3=data3["X"]
Yd3=data3["y"]

Xvald3=data3["Xval"]
Yvald3=data3["yval"]

#### Now we need to loop over C and sigma to find the best model for svm


def check_score():
    sigma=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    cost=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    score_initial=0

    for i in sigma:
        for j in cost:
            gamma=np.power(i,-2.)

            g_svm=svm.SVC(C=j,kernel='rbf',gamma=gamma)
            g_svm.fit(Xd3,Yd3.flatten())
            score=g_svm.score(Xvald3,Yvald3)
            if score>score_initial:
                score_initial=score
                pair=(j,i)

    return score_initial, pair
score, pair=check_score()

cost2=pair[0]
sigma2=pair[1]

#### First we plot data set 3 to see how it looks


c1d3=np.array([Xd3[i] for i in range(len(Xd3)) if Yd3[i]==1 ])
c2d3=np.array([Xd3[i] for i in range(len(Xd3)) if Yd3[i]==0])

c1X1d3=c1d3[:,0]
c1X2d3=c1d3[:,1]

c2X1d3=c2d3[:,0]
c2X2d3=c2d3[:,1]

def plot3():
    plt.figure(figsize=(10,6))
    plt.scatter(c1X1d3,c1X2d3,s=30,c='b',marker='x',linewidths=1)
    plt.scatter(c2X1d3,c2X2d3, s=30,c='r',marker='o',linewidths=1)
    plt.title("DataSet3")
plot3()


gamma2=np.power(sigma2,-2)
g_svm2=svm.SVC(C=cost2,kernel='rbf',gamma=gamma2)
g_svm2.fit(Xd3,Yd3.flatten())

def plot_svm3(svm):
    xvals = np.linspace(-0.6,0.2,100)
    yvals = np.linspace(-0.6,0.6,100)
    zvals = np.zeros((len(xvals),len(yvals)))
    for i in range(len(xvals)):
        for j in range(len(yvals)):
            zvals[i][j] = float(svm.predict(np.array([[xvals[i],yvals[j]]])))
    zvals = zvals.transpose()
    contour=plt.contour(xvals,yvals,zvals,[0])
    plt.title('Decision Boundary')


plot3()
plot_svm3(g_svm2)

### Now we move onto the spam classification section

#First load training and test data

stest=loadmat("/Users/rohitraikhy/Downloads/machine-learning-ex6 2/ex6/spamTest.mat")
strain=loadmat("/Users/rohitraikhy/Downloads/machine-learning-ex6 2/ex6/spamTrain.mat")

##Now we check to see the body of the data

def body(file):

    with open (file,"r") as f:
        e= f.read()
        return (e)
email=body("/Users/rohitraikhy/Downloads/machine-learning-ex6 3/ex6/emailSample1.txt")

###After seeing the body of the text the next step is to normalize the data

import re

###Lower case letters for the body
def lower(text):
    return email.lower()
email=lower(email)

###

test=re.compile(r'<[^>]+>')

def strip_html(text):
    return test.sub(' ', text)

email=strip_html(email)

#### Now we normalize URLS
def url(text):
    return re.sub(r'(http|https)://[^\s]*',"httpaddr", text)
email=url(email)

### Normalize email addresses
