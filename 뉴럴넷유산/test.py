import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
class Network:
    def __init__(self,Real_x,Real_y,hidden_size=100,total_layer=3):
        self.weight=[]
        self.each_layer_Dotproducted_values=[]
        self.bias=[]
        self.mask=[]
        self.masked_index=0
        self.layers=total_layer
        self.x=Real_x
        self.y=Real_y
        self.hidden_size=hidden_size
        self.dx=[]
        self.LearningRate=0.01
        for times in range(self.layers):
            if times==0:
                self.weight.append(np.random.randn(self.x.shape[1],self.hidden_size))#13,100
                self.bias.append(np.zeros(shape=hidden_size))
            elif times==self.layers-1:
                self.weight.append(np.random.randn(self.hidden_size,self.y.shape[1]))#100,2
                self.bias.append(np.zeros(shape=self.y.shape[1]))# 5로바꿨더니 forward 마지막층에서 broadcast에러가 남, 제대로 작동함.
            else:
                self.weight.append(np.random.randn(self.hidden_size,self.hidden_size))#100,100
                self.bias.append(np.zeros(self.hidden_size))
        print(np.average(self.x,axis=1))
    def forward(self):#sigmoid, leaky_relu 구성 Neural Network
        #최초 NN구성
        softmax=0
        for times in range(self.layers):
            if times==0:#최초 층
                temp=np.dot(self.x,self.weight[times])+self.bias[times]
                temp=self.Activation(temp)#leaky_relu
                #print('첫번째 층: ',temp[0:2],'\n')
                self.each_layer_Dotproducted_values.append(np.array(temp))
            elif times==1:#sigmoid 층 , or 연산자로 늘리기 가능.
                temp=np.dot(self.each_layer_Dotproducted_values[times-1],self.weight[times])+self.bias[times]
                temp=self.Activation(temp,type=1)#sigmoid
                self.each_layer_Dotproducted_values.append(np.array(temp))
            elif times==self.layers-1:#마지막 층
                temp=np.dot(self.each_layer_Dotproducted_values[times-1],self.weight[times])+self.bias[times]
                #temp=self.Activation(temp)
                print('마지막 층: ',temp[0:10],'\n','bias : ',self.bias,'\n')
                self.each_layer_Dotproducted_values.append(np.array(temp))
                softmax=self.softmax_cross(self.each_layer_Dotproducted_values[times])
                #argmax와 softmax의 결과 result
                #print('accuracy : ',self.accuracy(softmax))
                print('weight :',self.weight,'\n')
                #print(self.weight,'\n\n')
            else:#leaky_relu층
                temp=np.dot(self.each_layer_Dotproducted_values[times-1],self.weight[times])+self.bias[times]
                temp=self.Activation(temp)#leaky_relu
                #print('Leaky 층: ',temp[0:2],'\n')
                
                self.each_layer_Dotproducted_values.append(np.array(temp))
        #print(softmax,'\n\n')
        return softmax#이걸 하지 않으면 .. forward의 return 값이 없음으로 컴파일이 인식, 형식적인 return문.
        #최초 NN구성 END
        #loss func = > cross entropy
    def backward(self,softedmax):
        for back_times in range(self.layers):# self.dx는 역전파임으로 그대로 back_time을 사용, 이외 self.weight,bias는 순전파임으로 layers-1-back_times인덱스
            if back_times==0:#softmax _ cross entropy loss func
                soft_cross_back=softedmax-self.y#212,2
                self.dx.append(np.dot(soft_cross_back/self.y.shape[0],np.array(self.weight[(self.layers-1)]).T))#212,2 2,100 = 212,100
                self.weight[(self.layers-1)]-=self.LearningRate*(np.dot(self.each_layer_Dotproducted_values[(self.layers-2)].T,soft_cross_back))#212,2 212,2
                self.bias[(self.layers-1)]-=self.LearningRate*np.sum(soft_cross_back,axis=0)
            elif (self.layers-1)-back_times==1:#sigmoid backward,back_times=1
                temp=self.dx[back_times-1]*self.Activation(self.each_layer_Dotproducted_values[(self.layers-2)-back_times],type=1)*(1-self.Activation(self.each_layer_Dotproducted_values[(self.layers-2)-back_times],type=1))
                self.dx.append(np.dot(temp,np.array(self.weight[(self.layers-1)-back_times]).T))
                self.weight[(self.layers-1)-back_times]=self.weight[(self.layers-1)-back_times]-self.LearningRate*np.dot(np.array(self.each_layer_Dotproducted_values[(self.layers-2)-back_times]).T,temp)
                self.bias[(self.layers-1)-back_times]-=self.LearningRate*np.sum(temp,axis=0)
            elif (self.layers-1)-back_times==0:# 최초 층, back-timex==4
                temp=self.dx[back_times-1]
                temp[self.mask[self.masked_index]]*=0.1
                self.masked_index-=1
                self.weight[(self.layers-1)-back_times]=self.weight[(self.layers-1)-back_times]-self.LearningRate*np.dot(np.array(self.x.T),temp)
                self.bias[(self.layers-1)-back_times]-=self.LearningRate*np.sum(temp,axis=0)
            else:#leaky backward
                temp=self.dx[back_times-1]
                #print('1차:',temp,'\n','mask :',self.mask[self.masked_index-1],'\n')
                temp[self.mask[self.masked_index-1]]*=0.1
                #print('2차:',temp,'\n')
                #print(self.dx[back_times-1],temp[self.mask[self.masked_index-1]],self.mask[self.masked_index-1],'\n\n')
                self.masked_index-=1
                self.dx.append(np.dot(temp,np.array(self.weight[(self.layers-1)-back_times]).T))
                self.weight[(self.layers-1)-back_times]-=self.LearningRate*np.dot(np.array(self.each_layer_Dotproducted_values[(self.layers-2)-back_times]).T,temp)
                self.bias[(self.layers-1)-back_times]-=self.LearningRate*np.sum(temp,axis=0)
        self.dx.clear()
        self.each_layer_Dotproducted_values.clear()
                #self.weight[back_times]=self.weight[back_times]+self.LearningRate*()
        #위까지가 forward, 여기서부터 backpropagation
        
        #모두 완료되면, 기존에 있던 층의 곱값들을 모두 remove. 핵심은 가중치만 갱신하면됨.
    def Activation(self,input,type=0):
        #print(input,'Line')
        if type==0:
            mask=(input<=0.1)
            self.mask.append(mask)
            input[self.mask[self.masked_index]]*=0.1
            self.masked_index+=1
            #print(input,'\n\n')
            return input
        if type==1:#sigmoid
            #print(input)
            input=1/(1+np.exp(-input))
            return input
    def softmax_cross(self,input):
        input=np.exp(input)
        argmax=[]
        for i in range(len(input)):
            summation=np.sum(input[i])
            for j in range(len(input[i])):
                input[i][j]/=summation# summation을 여기에 갖다 넣는 것보다 변수로 변환시켜서 하는 게 확실함.
            argmax.append(np.argmax(input[i]))
        for i in range(len(input)):
            input[i]=-self.y[i]*np.log(input[i])
        #print(input)
        return input
    def accuracy(self,y_hat):
        y=np.argmax(self.y,axis=1)
        y_hat=np.argmax(y_hat,axis=1)
        #print(y[:30],'\n',y_hat[:30])
        return np.sum(y==y_hat)/len(y) 

# def standardization(data):#age,cp,trestbps,chol,thalach,oldpeak Stadardization
#     label=['age','cp','trestbps','chol','thalach','oldpeak']
#     for i in range(len(label)):
#         dvias=np.sqrt(np.sum((data[label[i]]-np.average(data[label[i]]))**2)/data[label[i]].shape[0]-10*np.exp(-7))
#         data[label[i]]=(data[label[i]]-np.average(data[label[i]]))/dvias
#     return data
# data=pd.read_csv('C:/Users/dlrms/OneDrive/Desktop/joljak/archive/heart.csv')
# data=standardization(data)
# # data shuffle
# # data standardization
# for i in range(data.shape[0]):
#     ran=np.random.randint(0,data.shape[0])
#     temp=data.iloc[ran]
#     data.iloc[ran]=data.iloc[i]
#     data.iloc[i]=temp

# x=data.drop(labels=['target'],axis=1)
# y=data['target']
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)#random 기능이 있어서 앞으로 위에 저거 할 필요는 없을듯

# x_train=np.array(x_train)
# y_train=pd.get_dummies(y_train)
# y_test=pd.get_dummies(y_test)
# y_train=np.array(y_train)
# x_test=np.array(x_test)
# y_test=np.array(y_test)
#y_train=y_train.reshape(y_train.shape[0],1)#앞으로 이 reshape가 굉장히 중요하다 ..(212,) 요런 새기들은 머신러닝의 악이야 시발.;.
data=pd.read_csv('C:/Users/dlrms/OneDrive/Desktop/Mnist/mnist_train.csv')
x=data.drop(labels=['label'],axis=1)
y=data['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)
y_train=pd.Series(y_train)
y_train=pd.get_dummies(data=y_train)
x_train=np.array(x_train)/255.0
y_train=np.array(y_train)
first=Network(x_train,y_train)

# for i in range(100):
#     kbs=first.forward()
#     first.backward(kbs)
    # print('\n\n',i,'\n\n')
