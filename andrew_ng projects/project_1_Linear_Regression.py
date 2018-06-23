import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)

### warm up exercise
def identity(n):
    return np.identity(n)
identity(5)

###### Linear Regression plotting the data

data=np.loadtxt('C:\Users\Rohit\Desktop\practice python\ex1data1.txt',delimiter=',')


x = np.c_[np.ones(data.shape[0]),data[:,0]]
y = np.c_[data[:,1]]

plt.scatter(x[:,1],y,s=30,c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000')
plt.ylabel('Profit in $10,000')

#### gradient descent -- cost function with theta initialized at 0,0

iterations=1500
alpha=0.01

m=y.size


def hyp(theta,X):
    return np.dot(X,theta)

def cost_function(Theta,x,y):
    m=y.size
    return float((1./(2*m))*np.dot((hyp(Theta,x)-y).T,(hyp(Theta,x)-y)))

start_theta=np.zeros((x.shape[1],1))
cost_function(start_theta,x,y)

###### gradient descenct algorythm


def gradient_descent(X,start_theta=np.zeros((2,1))):
    theta_values=[]
    cost_iterations=[]
    theta=start_theta
    for i in xrange(iterations):
        tmptheta=theta
        cost_iterations.append(cost_function(theta,X,y))
        theta_values.append(list(theta[:,0]))
        for j in xrange(len(tmptheta)):
            tmptheta[j]=theta[j]-(alpha/m)*np.dot((hyp(tmptheta,x)-y).T,np.array(x[:,j]).reshape(m,1))
        theta=tmptheta
    return theta, cost_iterations, theta_values
theta,cost_iterations, theta_values =gradient_descent(x,start_theta)

cost_iterations
theta_values
theta
cost_iterations
##### plotting cost function with respect to cost_iterations


def plot_cost(cost_iterations):
    plt.grid(True)
    plt.plot(cost_iterations)
    plt.ylim(4,7)
    plt.xlim(0,1500)
    plt.xlabel("iterations")
    plt.ylabel("Cost")

plot_cost(cost_iterations)


def linear_model(x_num):
    return (theta[0]+theta[1]*x_num)


plt.figure(figsize=(10,6))
plt.scatter(x[:,1],y,s=30,c='r', marker='x', linewidths=1)
plt.xlim(4,24)
plt.grid(True)
plt.xlabel('Population of City in 10,000')
plt.ylabel('Profit in $10,000')
plt.plot(x[:,1],linear_model(x[:,1]),label='linear_')



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
        zs.append(cost_function(np.array([[i],[j]]),x,y))

scat = ax.scatter(xs,ys,zs,c=np.abs(zs),cmap=plt.get_cmap('YlOrRd'))


plt.xlabel(r'$\theta_0$',fontsize=30)
plt.ylabel(r'$\theta_1$',fontsize=30)
plt.title('Cost (Minimization Path Shown in Blue)',fontsize=30)


plt.xlabel('theta0',fontsize=10)
plt.ylabel('theta 1',fontsize=10)
plt.title('Gradient Descent Graph',fontsize=20)
plt.plot([x[0] for x in theta_values],[x[1] for x in theta_values],cost_iterations,'bo-')
plt.show()



################## 2nd example of code



import numpy as np


B0=np.linspace(-10,10, num=50)
xx, yy = np.meshgrid(B0, B1, indexing='xy')
B1=np.linspace(-1,4, num =50)
Z = np.zeros((B0.size,B1.size))

for i in
    Z[i,j] = cost_function([[xx[i,j]], [yy[i,j]]],x,y)

fig = plt.figure(figsize=(15,6))
ax2 = fig.add_subplot(122, projection='3d')

ax2.plot_surface(xx, yy, Z, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
ax2.set_zlabel('Cost')
ax2.set_zlim(Z.min(),Z.max())
ax2.view_init(elev=15, azim=230)





###################  multivariate linear Regression
