# following codes are coded by Tek
# it's intended to realize MachineLearning function as trying to code it with base ML acknowledge
# it perform a image distribution  such Number like zero to nine

# If you have interset with my code or question simething about below code, contact me by email.
# Leekt970620@gmail.com


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
        if self.index==0:
            self.out.clear()
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
        if self.index==0:
            self.out.clear()
        return back
class Softmax_crossEntropy():
    def __init__(self,Realx,Realy):
        self.x=Realx
        self.y=Realy
        self.y_hat=None
        self.loss=None
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
        self.out=None
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
    def __init__(self,Realx,Realy,hidden_size_list,LR=0.05):# LR이 0.001일때는 거의 80%이하였는데..? 오 최초 80%돌파.. 
        self.x=Realx
        self.y=Realy
        self.LR=LR
        self.hidden_size_amount=len(hidden_size_list)
        self.hidden_weight_input_size=hidden_size_list
        self.propagation=[]
        self.backpropagation=[]
        self.weight=[]
        self.bias=[]
        self.total=0
        for i in range(self.hidden_size_amount):
            if i==0:
                self.weight.append(np.random.randn(self.x.shape[1],self.hidden_weight_input_size[i])*(1.0/np.sqrt(self.hidden_weight_input_size[i])))
                self.bias.append(np.zeros((self.hidden_weight_input_size[i])))
                print(self.hidden_weight_input_size[i])
            elif i==self.hidden_size_amount-1:
                self.weight.append(np.random.randn(self.hidden_weight_input_size[i-1],self.y.shape[1]))
                self.bias.append(np.zeros(self.y.shape[1]))
                print(self.hidden_weight_input_size[i-1])
            else:
                self.weight.append(np.random.randn(self.hidden_weight_input_size[i-1],self.hidden_weight_input_size[i])*(np.sqrt(2.0/self.hidden_weight_input_size[i])))
                self.bias.append(np.zeros((self.hidden_weight_input_size[i])))
                print(self.hidden_weight_input_size[i])
        self.sigmoid=Sigmoid()
        self.Relu=Relu()
        self.Softmax_cross=Softmax_crossEntropy(self.x,self.y)
        self.tanh=Tanh()
    def forward(self,batch_x,start,end,batch):#5개 층으로 구성해보자.
        self.batch_x=batch_x
        for i in range(self.hidden_size_amount):
            if i==0 or i==self.hidden_size_amount-3:
                dot=np.dot(batch_x,self.weight[i])+self.bias[i]
                activ_dot=self.Relu.forward(dot)
                self.propagation.append(activ_dot)
            elif i==self.hidden_size_amount-1:
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                loss,expectation=self.Softmax_cross.forward(dot,start,end,batch)
                return expectation
            elif i==self.hidden_size_amount-2:
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.tanh.forward(dot)
                self.propagation.append(activ_dot)
            else:
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.sigmoid.forward(dot)
                self.propagation.append(activ_dot)
    
    def test_forward(self,x_data,y_data):
        for i in range(self.hidden_size_amount):
            if i==0:
                dot=np.dot(x_data,self.weight[i])+self.bias[i]
                active_dot=self.Relu.forward(dot)
                self.propagation.append(active_dot)
            elif i==self.hidden_size_amount-1:
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                self.softmax_accuracy(dot,y_data)
                self.propagation.clear()
            else:
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.tanh.forward(dot)
                self.propagation.append(activ_dot)
    def backward(self):
        self.weight.reverse()
        self.bias.reverse()
        self.propagation.reverse()
        for i in range(self.hidden_size_amount):
            if i==0:
                back=self.Softmax_cross.backward()
                dx=np.dot(back,self.weight[i].T)
                dw=np.dot(self.propagation[i].T,back)
                db=np.sum(back,axis=0)
                self.weight[i]-=self.LR*dw
                self.bias[i]-=self.LR*db
                self.backpropagation.append(dx)
            elif i==self.hidden_size_amount-1:
                back=self.Relu.backward(self.backpropagation[i-1])
                dw=np.dot(self.batch_x.T,back)
                db=np.sum(back,axis=0)
                self.weight[i]-=self.LR*dw
                self.bias[i]-=self.LR*db
                self.weight.reverse()
                self.bias.reverse()
                self.propagation.clear()
                self.backpropagation.clear()
            else:
                back=self.tanh.backward(self.backpropagation[i-1])
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
# data=pd.read_csv('C:/Users/dlrms/OneDrive/Desktop/Mnist/mnist_train.csv')
test_data=pd.read_csv("C:/Users/dlrms/OneDrive/Desktop/file3.csv",index_col=0)
test_x_data=test_data.drop(labels=["lable","name"],axis=1)
test_y_data=test_data['lable']
# test_y_data=np.array(pd.get_dummies(data=test_y_data))
# print(test_y_data)
# test_x_data=np.array(test_x_data/255.0)
# x=data.drop(labels=['label'],axis=1)
# y=data['label']
x_train,x_test,y_train,y_test=train_test_split(test_x_data,test_y_data,test_size=0.2,random_state=123)
#print(len(x_train),len(x_test)) # 48000 12000 , 73 19
# y_train=pd.Series(y_train)
y_train=pd.get_dummies(data=y_train)
y_train=np.array(y_train)
x_train=np.array(x_train/255.0)
x_test=np.array(x_test/255.0)
y_test=pd.get_dummies(y_test)
y_test=np.array(y_test)
NeuralNet= Main(x_train,y_train,[3000,2050,1000])
#data shuffle
# for i in range(data.shape[0]):
#     ran=np.random.randint(0,data.shape[0])
#     temp=data.iloc[ran]
#     data.iloc[ran]=data.iloc[i]
#     data.iloc[i]=temp
epoch=6
# validation=1000
flag=0
batch=300#int((len(x_train))/50)
for epochs in range(0,epoch+1):
    for batchs in range(0,len(x_train),batch):
        #print(len(x_train[batchs:batchs+batch]))73개
        forward=NeuralNet.forward(x_train[batchs:batchs+batch],start=batchs,end=batchs+batch,batch=batch)
        back=NeuralNet.backward()
    if epochs==0 or epochs==1 or epochs%10==0:
        print("epoch : ", epochs)
        print("accuracy : ", forward)
# 즉 batch 안에서도..~ 만약 1만개 배치면? 내가만약 이 안에서 10번 더 나눠서 넣을거다? 그러면 0+1000, 1+1000 ... 이렇게 하면 됨.
NeuralNet.test_forward(x_test,y_test)
# for test in range(0,len(np.array(x_train)),validation):
#     if (test+validation)>=len(x_train): # 이거 추가하니까 더 이상 ..! 문제가 발생하지 않음. 아무래도 1씩 증가하니까 이렇게 되는 듯.
#         break;
#     x_validation,y_validation,x_training,y_training=pandas_Validation(x_train,y_train,test=test,validation=validation)
#     if flag==0:
#         NeuralNet= Main(x_training,y_training,[850,850,600])
#         #Main 클래스가 다시 실행되면 np random이 다시 만들어져서 이전에 했던 훈련은 무용지물이 됨. 그래서 이건 딱 한 버만 선언되어야 함.
#         flag+=1
#     for i in range(epoch+1):#학습시킬 총횟수.(반복학습을 얼마나 시킬것인가)
#         for j in range(0,len(x_training),batch):#학습량중에서 몇개씩 나눠서 학습할 것인지 batch.
#             accuracy=NeuralNet.forward(batch_x=x_training[j:batch+j],start=j,end=batch+j,batch=batch,epoch=i)
#             print("pass")
#             back=NeuralNet.backward()
#         if i==0 or i==1 or i%100==0:
#             print('epoch : ',i)
#             print('accuracy : ',accuracy)
#     #여기서부터 실제 validation으로 체크.
#     for j in range(0,len(x_validation),len(x_validation)):
#         print(j,'~',len(x_validation),"Validation test_accuracy")
#         kbs=NeuralNet.test_forward(x_validation[j:len(x_validation)+j],y_validation[j:len(x_validation)+j])
    
#     for c in range(0,len(x_test),len(x_test)):
#         print(c,'~',c+len(x_test),"Real test_accuracy")
#         kbs=NeuralNet.test_forward(x_test[c:len(x_test)+c],y_test[c:len(x_test)+c])
#128 128 128 의 테스트 결과 48%, 오버피팅 문제인 듯. lr = 0.01

# 1000 512 128 로 시도, lr:0.005 - >  