import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
class input_layer:
    def __init__(self,input):
        self.input=input
        self.w=np.random.randn(self.input.shape[1],1000) # 784, 100
        self.bias=np.random.random(1000)
        self.dw=None
        self.dbias=None
    def forward(self):# batch만큼 들어옴.
        output=np.dot(self.input,self.w)+self.bias
        output=self.Active_LeakyRelu(output)
        #self.mask=(output<=0)
        return output
    def backward(self,output):# dL/d? = d(Loss)/d(Relu)*d(Relu)/d(input)=d(Loss)/d(Relu)*1(x>0),*0.1(x<=0)
        #Leakyrelu 의 특징 x를 미분하면 1, 0.1x을 미분하면 0.1
        out=1#최종출력단
        for i in range(len(output)):
            for j in range(len(output[0])):
                if output[i][j]<=0:
                    output[i][j]=0.1*output[i][j]
                #else: output[i][j]=1*output[i][j]
        self.dw=np.dot(self.input.T,output)#
        self.dbias=np.sum(output,axis=0)
        self.updata()
        #dx는 없음. 첫번째 단이므로.
    def updata(self):
        self.w=self.w-0.01*self.dw
        self.bias-=0.01*self.dbias
    def Active_LeakyRelu(self,output):
        for i in range(len(output)):
            for j in range(len(output[0])):
                if output[i][j]<0:
                    output[i][j]*=0.1
        return output

class hidden_first:
    def __init__(self,y):
        self.input=None
        self.y=y
        self.w=np.random.randn(1000,self.y.shape[1])
        self.bias=np.random.random(y.shape[1])
        self.output=None
        self.dw=None
        self.dbias=None
    def forward(self,input):
        self.input=input
        output=np.dot(self.input,self.w)+self.bias
        #print(output.shape)
        self.output=self.softmax(output)
        loss=self.loss(self.output)
        accuracy=self.accuracy(self.output,self.y)
        print(accuracy)
    def backward(self):
        batch=self.y.shape[0] # batch는 도대체 왜 나눠야 하나..
        input=(self.output-self.y)/batch
        self.dw=np.dot(self.input.T,input)
        self.dbias=np.sum(input,axis=0)
        dx=np.dot(input,self.w.T)
        self.updata()
        return dx
    def updata(self):
        self.w=self.w-0.01*self.dw
        self.bias-=0.01*self.dbias
    def softmax(slef,output): # 울프람 결과 ..작동은 전체 합은 1로 잘 작동한다.
        for i in range(output.shape[0]):
            max=np.max(output[i])
            output[i]=np.exp(output[i]-max)
            output[i]/=np.sum(output[i])
        return output
    def loss(self,output):# 1인 것만 곱해서 log취하면 잘 된다.log 안에 소수점이 많아질수록 그 큰값이 나온다.
        #cross_entropy
        output=-np.sum(self.y*np.log(output),axis=1)
        return output
    def accuracy(self,y_hat,y):
        y=np.argmax(y,axis=1)
        y_hat=np.argmax(y_hat,axis=1)
        print(y_hat[:10])
        return np.sum(y==y_hat)/len(y)
        

start1=time.time()
data=pd.read_csv('C:/Users/dlrms/OneDrive/Desktop/archive (1)/mnist_train.csv')
x=data.drop(labels=['label'],axis=1)
y=data['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)
y_train=pd.Series(y_train)
y_train=pd.get_dummies(data=y_train)
#print(np.array(x_train),'\n',y_train)
first=input_layer(x_train/255)
second=hidden_first(np.array(y_train))
end1=time.time()
#print('데이터 전처리 단계 시간',end1-start1,'\n')
batch_size=100
start=time.time()
for i in range(100):
    #input_data=np.array(x_train)[:batch_size]/255
    #start2=time.time()
    second.forward(first.forward())
    #end2=time.time()
    #print('forward시간 : ',end2-start2,'\n')
    #start3=time.time()
    first.backward(second.backward())
    #print('backward 시간 :',time.time()-start3,'\n')
    print('epoch :',i,'\n')
print(time.time()-start)
