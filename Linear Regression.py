# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:24:28 2021

@author: usar
"""

import pandas as pd
import numpy as np
import array as arr
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/usar/Desktop/p/machine learning/headbrain.csv")
data.head(7)

x=data['Head Size(cm^3)'].values
y=data['Brain Weight(grams)'].values

# dir(np)
mean_x=np.mean(x)
mean_y=np.mean(y)

m=len(x)
num=0
den=0
for i in range(m):
    num+=(x[i]-mean_x)*(y[i]-mean_y)
    den+=(x[i]-mean_x)**2
    
m=num/den
c=mean_y-mean_x*m
print('slope = ',m,'and intercept = ',c)

#plotting
X=x
Y=m*X + c

plt.plot(X,Y,color='green',label='linear regression line')
plt.scatter(x,y,color='red',label='Scatter plot')
plt.legend()
plt.xlabel('HEAD SIZE')
plt.ylabel('BRAIN WEIGHT')
plt.title('LINEAR REGRESSION')
plt.grid(color='blue')
plt.show()

#

# while True:
#     inp=int(input('ENTER HEAD SIZE '))
#     ou=m*inp + c
#     print( 'for input head size of ',inp,' the weight of brain is ',ou)



# #goodness of line
# ss_t=0
# ss_r=0
# for i in range(m):
#     ss_t+=(Y[i]-mean_y)**2
#     ss_r+=(y[i]-mean_y)**2
    
# r2=ss_t/ss_r
# print(r2)

#########################################EXAMPLE##############


#########################################LINEAR REGRESSSION USING SKLEARN
X1=x.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X1,y,test_size=1/4,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)


plt.plot(X_train,reg.predict(X_train),color='red',label='training line')
plt.plot(X_test,reg.predict(X_test),color='green',label='testing line')
plt.scatter(x,y,color='orange',label='scatter')
plt.show()


