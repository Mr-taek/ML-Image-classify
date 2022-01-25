import numpy as np 

class Tanh():
    def __init__(self):
        self.out=[]
        self.index=0
    def forward(self,input):
        self.out.append((np.exp(input)-np.exp(-input))/(np.exp(input)+np.exp(-input)))
        self.index+=1
        return self.out[self.index-1]
    def backward(self,backpropagation):
        dx=backpropagation*(1-self.out[self.index-1]*self.out[self.index-1])
        self.index-=1
        return dx
class Sigmoid():
    def __init__(self):
        self.out=[]
        self.index=0
    def forward(self,input):
        self.out.append(1/(1+np.exp(-input)))
        self.index+=1
        return self.out[self.index-1]
    def backward(self,backpropagation):
        back=backpropagation*self.out[self.index-1]*(1-self.out[self.index-1])
        self.index-=1
        return back
class Softmax_crossEntropy():
    def __init__(self,Realx,Realy):
        self.x=Realx
        self.y=Realy
        self.y_hat=None
        self.loss=None
        self.start=0
        self.end=0
        self.batch=0
    def forward(self,input,start,end,batch):
        self.batch=batch
        self.start=start
        self.end=end
        input=input.T
        max=np.max(input,axis=0)
        self.input=input-max
        self.y_hat=np.exp(self.input)/np.sum(np.exp(self.input),axis=0)
        self.y_hat=self.y_hat.T
        expectation=self.accuracy(self.y_hat)
        self.loss=self.cross_entropy(self.y_hat)
        return self.loss,expectation
    def accuracy(self,y_hat):
        maxy=np.argmax(self.y[self.start:self.end],axis=1)
        maxy_hat=np.argmax(y_hat,axis=1)
        return np.sum(maxy==maxy_hat)/maxy.shape[0]
    def cross_entropy(self,input):
        cross=np.sum(self.y[self.start:self.end]*np.log(input+1e-7),axis=1)
        entropy=-np.sum(cross)/self.batch
        #print(entropy)
        return entropy
    def backward(self,backpropagation=1.0):
        back=backpropagation*(self.y_hat-self.y[self.start:self.end])/self.batch
        return back
class Relu():
    def __init__(self):
        self.mask=[]
        self.idx=0
        #self.out=None#이거 없어서 오류 났는데 이제는 없애돟 또 안 남..
    def forward(self,input):
        self.mask.append((input<=0))
        self.out=input.copy()
        self.out[input<=0]*=0
        self.idx+=1
        return self.out
    def backward(self,backpropagation):
        backpropagation[self.mask[self.idx-1]]*=0
        self.idx-=1
        dx=backpropagation
        return dx
class Main():
    def __init__(self,Realx,Realy,hidden_size_list,LR=0.01):# LR이 0.001일때는 거의 80%이하였는데..? 오 최초 80%돌파.. 
        self.x=Realx
        self.y=Realy
        self.LR=LR
        self.hidden_size_amount=len(hidden_size_list) # +1은 다른 for문에서 사용될 때 인덱스 -1을 인지하여 +1한 것.
        self.hidden_weight_input_size=hidden_size_list
        self.propagation=[]
        self.backpropagation=[]
        self.weight=[]
        self.bias=[]
        self.total=0
        for i in range(self.hidden_size_amount+1):
            if i==0:#r
                self.weight.append(np.random.randn(self.x.shape[1],self.hidden_weight_input_size[i])*(1.0/np.sqrt(self.hidden_weight_input_size[i])))
                self.bias.append(np.zeros(self.hidden_weight_input_size[i]))
            elif i==1:#t
                self.weight.append(np.random.randn(self.hidden_weight_input_size[i-1],self.hidden_weight_input_size[i]))
                self.bias.append(np.zeros(self.hidden_weight_input_size[i]))
            elif i==2:#r
                self.weight.append(np.random.randn(self.hidden_weight_input_size[i-1],self.hidden_weight_input_size[i])*(1.0/np.sqrt(self.hidden_weight_input_size[i])))
                self.bias.append(np.zeros(self.hidden_weight_input_size[i]))
            elif i==3:#s
                self.weight.append(np.random.randn(self.hidden_weight_input_size[i-1],self.hidden_weight_input_size[i])*(np.sqrt(2.0/self.hidden_weight_input_size[i])))
                self.bias.append(np.zeros(self.hidden_weight_input_size[i]))
            elif i==4:
                self.weight.append(np.random.randn(self.hidden_weight_input_size[i-1],self.hidden_weight_input_size[i])*(1.0/np.sqrt(self.hidden_weight_input_size[i])))
                self.bias.append(np.zeros(self.hidden_weight_input_size[i]))
            elif i==5:
                self.weight.append(np.random.randn(self.hidden_weight_input_size[i-1],self.hidden_weight_input_size[i]))
                self.bias.append(np.zeros(self.hidden_weight_input_size[i]))
            elif i==6:
                self.weight.append(np.random.randn(self.hidden_weight_input_size[i-1],self.hidden_weight_input_size[i])*(1.0/np.sqrt(self.hidden_weight_input_size[i])))
                self.bias.append(np.zeros(self.hidden_weight_input_size[i]))
            elif i==7:
                self.weight.append(np.random.randn(self.hidden_weight_input_size[i-1],self.hidden_weight_input_size[i])*(np.sqrt(2.0/self.hidden_weight_input_size[i])))
                self.bias.append(np.zeros(self.hidden_weight_input_size[i]))
            elif i==8:
                self.weight.append(np.random.randn(self.hidden_weight_input_size[i-1],self.y.shape[1]))
                self.bias.append(np.zeros(self.y.shape[1]))
        self.sigmoid=Sigmoid()
        self.Relu=Relu()
        self.Softmax_cross=Softmax_crossEntropy(self.x,self.y)
        self.tanh=Tanh()
    def forward(self,batch_x,start,end,batch):#5개 층으로 구성해보자.
        self.batch_x=batch_x
        for i in range(self.hidden_size_amount+1):
            if i==0 : 
                dot=np.dot(batch_x,self.weight[i])+self.bias[i]
                activ_dot=self.Relu.forward(dot)
                self.propagation.append(activ_dot)
            elif i==1:
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.tanh.forward(dot)
                self.propagation.append(activ_dot)
            elif i==2: 
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.Relu.forward(dot)
                self.propagation.append(activ_dot)
            elif i==3: 
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.sigmoid.forward(dot)
                self.propagation.append(activ_dot)
            elif i==4 : 
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.Relu.forward(dot)
                self.propagation.append(activ_dot)
            elif i==5:# 
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.tanh.forward(dot)
                self.propagation.append(activ_dot)
            elif i==6: 
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.Relu.forward(dot)
                self.propagation.append(activ_dot)
            elif i==7:
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.sigmoid.forward(dot)
                self.propagation.append(activ_dot)
            elif i==self.hidden_size_amount:
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                loss,expectation=self.Softmax_cross.forward(dot,start,end,batch)
                return expectation
            
    
    def test_forward(self,x_data,y_data):
        for i in range(self.hidden_size_amount+1):
            # if i==0 :#1. relu
            #     dot=np.dot(x_data,self.weight[i])+self.bias[i]
            #     active_dot=self.Relu.forward(dot)
            #     self.propagation.append(active_dot)
            # elif i==1:# 4 tanh
            #     dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
            #     activ_dot=self.tanh.forward(dot)
            #     self.propagation.append(activ_dot)
            # elif i==2: # 3relu
            #     dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
            #     activ_dot=self.Relu.forward(dot)
            #     self.propagation.append(activ_dot)
            # elif i==3:#2. sigmoid
            #     dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
            #     activ_dot=self.sigmoid.forward(dot)
            #     self.propagation.append(activ_dot)
            # elif i==4 :#1. relu
            #     dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
            #     activ_dot=self.Relu.forward(dot)
            #     self.propagation.append(activ_dot)
            # elif i==5:# 4 tanh
            #     dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
            #     activ_dot=self.tanh.forward(dot)
            #     self.propagation.append(activ_dot)
            # elif i==6: # 3relu
            #     dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
            #     activ_dot=self.Relu.forward(dot)
            #     self.propagation.append(activ_dot)
            # elif i==7:#2. sigmoid
            #     dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
            #     activ_dot=self.sigmoid.forward(dot)
            #     self.propagation.append(activ_dot)
            # elif i==self.hidden_size_amount:#softmax
            #     dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
            #     self.softmax_accuracy(dot,y_data)
            #     self.propagation.clear()
            if i==0 : 
                dot=np.dot(x_data,self.weight[i])+self.bias[i]
                activ_dot=self.Relu.forward(dot)
                self.propagation.append(activ_dot)
            elif i==1:
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.tanh.forward(dot)
                self.propagation.append(activ_dot)
            elif i==2: 
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.Relu.forward(dot)
                self.propagation.append(activ_dot)
            elif i==3: 
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.sigmoid.forward(dot)
                self.propagation.append(activ_dot)
            elif i==4 : 
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.Relu.forward(dot)
                self.propagation.append(activ_dot)
            elif i==5:# 
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.tanh.forward(dot)
                self.propagation.append(activ_dot)
            elif i==6: 
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.Relu.forward(dot)
                self.propagation.append(activ_dot)
            elif i==7:
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.sigmoid.forward(dot)
                self.propagation.append(activ_dot)
            elif i==self.hidden_size_amount:#softmax
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                self.softmax_accuracy(dot,y_data)
                self.propagation.clear()
            
    def backward(self):
        self.weight.reverse()
        self.bias.reverse()
        self.propagation.reverse()
        for i in range(self.hidden_size_amount+1):
            if i==self.hidden_size_amount:
                back=self.Relu.backward(self.backpropagation[i-1])
                dw=np.dot(self.batch_x.T,back)
                db=np.sum(back,axis=0)
                self.weight[i]-=self.LR*dw
                self.bias[i]-=self.LR*db
                self.weight.reverse()
                self.bias.reverse()
                self.propagation.clear()
                self.backpropagation.clear()
            elif i==7:#1
                back=self.tanh.backward(self.backpropagation[i-1])
                dx=np.dot(back,self.weight[i].T)
                dw=np.dot(self.propagation[i].T,back)
                db=np.sum(back,axis=0)
                self.weight[i]-=self.LR*dw
                self.bias[i]-=self.LR*db
                self.backpropagation.append(dx)
            elif i==6:# 2
                back=self.Relu.backward(self.backpropagation[i-1])
                dx=np.dot(back,self.weight[i].T)
                dw=np.dot(self.propagation[i].T,back)
                db=np.sum(back,axis=0)
                self.weight[i]-=self.LR*dw
                self.bias[i]-=self.LR*db
                self.backpropagation.append(dx)
            elif i==5:#3
                back=self.sigmoid.backward(self.backpropagation[i-1])
                dx=np.dot(back,self.weight[i].T)
                dw=np.dot(self.propagation[i].T,back)
                db=np.sum(back,axis=0)
                self.weight[i]-=self.LR*dw
                self.bias[i]-=self.LR*db
                self.backpropagation.append(dx)
            elif i==4:
                back=self.Relu.backward(self.backpropagation[i-1])
                dx=np.dot(back,self.weight[i].T)
                dw=np.dot(self.propagation[i].T,back)
                db=np.sum(back,axis=0)
                self.weight[i]-=self.LR*dw
                self.bias[i]-=self.LR*db
                self.backpropagation.append(dx)
            elif i==3:#1
                back=self.tanh.backward(self.backpropagation[i-1])
                
                dx=np.dot(back,self.weight[i].T)
                dw=np.dot(self.propagation[i].T,back)
                db=np.sum(back,axis=0)
                self.weight[i]-=self.LR*dw
                self.bias[i]-=self.LR*db
                
                self.backpropagation.append(dx)
            elif i==2:# 2
                back=self.Relu.backward(self.backpropagation[i-1])
                
                dx=np.dot(back,self.weight[i].T)
                dw=np.dot(self.propagation[i].T,back)
                db=np.sum(back,axis=0)
                self.weight[i]-=self.LR*dw
                self.bias[i]-=self.LR*db
                
                self.backpropagation.append(dx)
            elif i==1:#3
                back=self.sigmoid.backward(self.backpropagation[i-1])
                dx=np.dot(back,self.weight[i].T)
                dw=np.dot(self.propagation[i].T,back)
                db=np.sum(back,axis=0)
                self.weight[i]-=self.LR*dw
                self.bias[i]-=self.LR*db
                self.backpropagation.append(dx)
            elif i==0:#0
                back=self.Softmax_cross.backward()
                dx=np.dot(back,self.weight[i].T)
                dw=np.dot(self.propagation[i].T,back)
                db=np.sum(back,axis=0)
                self.weight[i]-=self.LR*dw
                self.bias[i]-=self.LR*db
                self.backpropagation.append(dx)
            
            
    def softmax_accuracy(self,output,y_data):
        output=output.T
        max=np.max(output,axis=0)
        output=np.exp(output-max)
        output=(output/np.sum(output,axis=0)).T
        yhat_argmax=np.argmax(output,axis=1)
        y_argmax=np.argmax(y_data,axis=1)
        res=np.sum(yhat_argmax==y_argmax)/y_data.shape[0]
        print("test_accuracy : ",res,'\n')


import pandas as pd
from sklearn.model_selection import train_test_split
def pandas_Validation(x,y,test,validation):
    x_validation=x.iloc[test:test+validation][:-1]
    y_validation=y.iloc[test:test+validation][:-1]#원래는 y_train.index 여서 pandas 파일이었나봄.
    x_training=x.drop(labels=x_validation.index,axis=0)
    y_training=y.drop(labels=y_validation.index,axis=0)
    x_validation=np.array(x_validation)/255.0
    x_training=np.array(x_training)/255.0
    y_validation=pd.get_dummies(y_validation)
    y_validation=np.array(y_validation)
    y_training=pd.get_dummies(y_training)
    y_training=np.array(y_training)
    return x_validation,y_validation,x_training,y_training

test_data=pd.read_csv("C:/Users/dlrms/OneDrive/Desktop/file3.csv",index_col=0)
test_x_data=test_data.drop(labels=["lable","name"],axis=1)
test_y_data=test_data['lable']

x_train,x_test,y_train,y_test=train_test_split(test_x_data,test_y_data,test_size=0.2,random_state=123)

y_train=pd.get_dummies(data=y_train)
y_train=np.array(y_train)
x_train=np.array(x_train/255.0)
x_test=np.array(x_test/255.0)
y_test=pd.get_dummies(y_test)
y_test=np.array(y_test)
NeuralNet= Main(x_train,y_train,[2000,1500,1000,700,400,300,200,100])

epoch=200
flag=0
batch=300
for epochs in range(0,epoch+1):
    for batchs in range(0,len(x_train),batch):
        #print(len(x_train[batchs:batchs+batch]))73개
        forward=NeuralNet.forward(x_train[batchs:batchs+batch],start=batchs,end=batchs+batch,batch=batch)
        back=NeuralNet.backward()
    if epochs==0 or epochs==1 or epochs%10==0:
        print("epoch : ", epochs)
        print("accuracy : ", forward)

NeuralNet.test_forward(x_test,y_test)
