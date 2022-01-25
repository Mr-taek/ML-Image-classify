#행에 대한 값만 loss에 들어오면 for은 한 번만 쓰면 됨.
def loss(self,y_hat,y):# numpy array
    loss=(y_hat-y)**2 # 1차원 - 1차원 
    sum=0
    for i in range(len(y_hat)):# 1차원에 대한 len은 원소의 개수.
        sum+=loss[i]
    return 0.5*sum

import numpy as np
a=np.array([[-1,2,-3,43,5],[-5,5,35,23,-1]])

b=np.random.random((2,2))

#x=np.array([[1,2,3],[2,3,4]])+np.array([1,2])
#print(x)
c=np.random.random(3)
x=np.array([[1,2,3],[2,3,4]])
#print(np.exp(x))
import pandas as pd
t=[[0,1,0],[1,0,0],[0,1,0]]
t=np.argmax(t,axis=1)
y=np.array([[0.1,0.8,0.78],[0.8,0.78,0.45],[0.34,0.26,0.89]])
y=np.argmax(y,axis=1)
s=np.sum(y==t)/len(y)
t=np.array([0.6,0.98])
high=0
y=np.array([[0.1,0.8],[0.8,0.78],[0.34,0.26]])
print(y*[[True],[False,True],[True]])

