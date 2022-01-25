import numpy as np
from numpy.lib.function_base import average
import pandas as pd
from sklearn.model_selection import train_test_split
# 모멘텀 optimizer을 사용. 단 은닉층 구성.
# batch크기가 클 수록 한번에 학습 하는 양이 많아서 '가중치의 타 데이터에 대한 보편성'이 증가. 동시에 Learning rate을 함께 조정해야 함.

class one:
    def __init__(self,True_x,True_y):
        self.w=np.random.randn(True_x.shape[1],True_y.shape[1])# y 데이터는 1열이라.. (212,) 으로 등장.
        self.velocity=np.zeros(self.w.shape)
        self.accuracy_before=80.0#최초 최소 정확도
    def main(self,x_,y_,Type=0):
        if Type==0:
            output=np.dot(x_,self.w)
            self.accuracy_now=self.accuracy(y_,output)
            if self.accuracy_before>self.accuracy_now:
                output=self.Relu(output) # Relu를 사용하지 않으면 거의 평균 50% 의 accuracy
                momen=0.9*self.velocity-0.01*self.grad(output,x_,y_,Type=0)
                output=self.softmax(output)
                self.w=self.w+momen # 여기를 - 로해서 계속 문제가 발생.
            self.accuracy_before=self.accuracy_now
            print('train accuracy : ',self.accuracy_before,'\n')
        elif Type==1:
            output=np.dot(x_,self.w)
            output=self.softmax(output)
            print('test accuracy : ',self.accuracy(y_,output),'\n')
    def grad(self,output,x_,y_,Type=0):
        #Type=0 : mean_squre ,=1 : cross_entropy
        if Type==1:
            return np.dot(x_.T,(output-y_))# 실제로 0.75에서 증가하지 않는 gradient vanishing을 맞이함.. 
        elif Type==0:
            return np.dot(x_.T,y_*(1/np.log(output)))
    def Relu(self,output):
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if output[i][j]<=0:
                    output[i][j]=0
        return output
    def softmax(self,output):
        #print(np.sum(output,axis=1))
        for i in range(output.shape[0]):
            max=np.max(output[i])
            output[i]=np.exp(output[i])/np.sum(output[i],axis=0)
        return output
    def accuracy(self,y_,y_hat):
        y_=np.argmax(y_,axis=1)
        #print(y_hat)
        y_hat=np.argmax(y_hat,axis=1)
        #print(y_,'\n',y_hat)
        return np.sum(y_==y_hat)/y_.shape[0]


def standardization(data):#age,cp,trestbps,chol,thalach,oldpeak Stadardization
    label=['age','cp','trestbps','chol','thalach','oldpeak']
    for i in range(len(label)):
        dvias=np.sqrt(np.sum((data[label[i]]-np.average(data[label[i]]))**2)/data[label[i]].shape[0]-10*np.exp(-7))
        data[label[i]]=(data[label[i]]-np.average(data[label[i]]))/dvias
    return data
data=pd.read_csv('C:/Users/dlrms/OneDrive/Desktop/joljak/archive/heart.csv')
data=standardization(data)
# data shuffle
# data standardization
for i in range(data.shape[0]):
    ran=np.random.randint(0,data.shape[0])
    temp=data.iloc[ran]
    data.iloc[ran]=data.iloc[i]
    data.iloc[i]=temp

x=data.drop(labels=['target'],axis=1)
y=data['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)#random 기능이 있어서 앞으로 위에 저거 할 필요는 없을듯
x_train=np.array(x_train)
y_train=pd.get_dummies(y_train)
y_test=pd.get_dummies(y_test)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)
#y_train=y_train.reshape(y_train.shape[0],1)#앞으로 이 reshape가 굉장히 중요하다 ..(212,) 요런 새기들은 머신러닝의 악이야 시발.;.
first=one(x_train,y_train)
batch=1000

for i in range(100):
    for j in range(0,len(x_train),batch):
        first.main(x_train,y_train)