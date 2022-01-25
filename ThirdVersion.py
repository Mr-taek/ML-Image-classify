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
        if self.index==0:#안하면.. 잔재가 남아있게 돼서 test에 accuracy에 덮어 씌우게 됨
            self.out.clear()
        return back
class Softmax_crossEntropy():
    def __init__(self,Realx,Realy):
        self.x=Realx
        self.y=Realy
        # self.y_hat=None
        # self.loss=None
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
        # print(cross)
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
        if self.idx==0:
            self.mask.clear()
        return dx
class Main():
    def __init__(self,Realx,Realy,hidden_size_list,Node_list,LR=0.01):# LR이 0.001일때는 거의 80%이하였는데..? 오 최초 80%돌파.. 
        self.x=Realx
        self.y=Realy
        self.LR=LR
        self.hidden_size_amount=len(hidden_size_list)-1 # -1은 노드가 총 3개면 0,1,2 임으로 2까지 만들기 위함. node_type으로 for문을 돌리기 때문에
        #마지막 노드를 인식하려면 이 변수가 노드 리스트의 길이의 -1 되어야 함.
        self.hidden_weight_input_size=hidden_size_list
        self.node_type=Node_list # 'Relu' 'Tahh' 'Relu'
        self.propagation=[]
        self.backpropagation=[]
        self.weight=[]
        self.bias=[]
        self.total=0
        for index,node in enumerate(hidden_size_list):# 0번 노드 (x,2000) 1.(2000,1000) 마지막노드(2). (1000, y)
            if index==0: # 1번노드
                self.weight.append(np.random.randn(self.x.shape[1],node)*(np.sqrt(2.0/node)))
                self.bias.append(np.zeros(node))
            elif index==self.hidden_size_amount: # 마지막노드
                self.weight.append(np.random.randn(hidden_size_list[index-1],node)*(np.sqrt(2.0/node)))#(np.sqrt(2.0/node))
                self.bias.append(np.zeros(node))
                self.weight.append(np.random.randn(node,self.y.shape[1]))
                self.bias.append(np.zeros(self.y.shape[1]))
            else :# n번노드
                self.weight.append(np.random.randn(hidden_size_list[index-1],node)*(np.sqrt(2.0/node)))#(1.0/np.sqrt(node))
                self.bias.append(np.zeros(node))
        self.sigmoid=Sigmoid()
        self.Relu=Relu()
        self.Softmax_cross=Softmax_crossEntropy(self.x,self.y)
        self.tanh=Tanh()
    def forward(self,batch_x,start,end,batch):#5개 층으로 구성해보자.
        self.batch_x=batch_x
        for i,name in enumerate(self.node_type):
            if i==0:
                self.propagation.append(self.Node_box('f',name=name,index=i,data=batch_x))
            elif i==self.hidden_size_amount:#softmax는 건들지 말자.
                self.propagation.append(self.Node_box('f',name,i,self.propagation[i-1]))# 169,1000
                dot=np.dot(self.propagation[i],self.weight[i+1])+self.bias[i+1]#169,1000 dot 1000, y + y
                #self.propagation.append(dot)# 169,9 이건 안 하는게 ...
                loss,expectation=self.Softmax_cross.forward(dot,start,end,batch)
                return expectation
            else:
                self.propagation.append(self.Node_box('f',name,i,self.propagation[i-1]))
    def test_forward(self,x_data,y_data):
        for i,name in enumerate(self.node_type):
            if i==0:
                self.propagation.append(self.Node_box('f',name=name,index=i,data=x_data))
            elif i==self.hidden_size_amount:#softmax
                self.propagation.append(self.Node_box('f',name,i,self.propagation[i-1]))# 169,1000
                #print(self.propagation[i-1].shape)
                dot=np.dot(self.propagation[i],self.weight[i+1])+self.bias[i+1]#169,1000 dot 1000, y + y
                #print(self.propagation[i].shape,dot.shape)
                diff_index,pred_num,real_num=self.softmax_accuracy(dot,y_data,x_data)
                self.propagation.clear()
                return diff_index,pred_num,real_num
            else : 
                self.propagation.append(self.Node_box('f',name,i,self.propagation[i-1]))
                
    def softmax_accuracy(self,output,y_data,x_data):
        #print(output.shape,y_data.shape)
        output=output.T
        max=np.max(output,axis=0)
        output=np.exp(output-max)
        output=(output/np.sum(output,axis=0)).T
        yhat_argmax=np.argmax(output,axis=1)
        y_argmax=np.argmax(y_data,axis=1)
        #print("예측 숫자 : ",yhat_argmax,'\n','실제 숫자 : ',y_argmax)
        res=np.sum(yhat_argmax==y_argmax)/y_data.shape[0]
        diff_index=[]
        pred_num=[]
        real_num=[]
        for i in range(len(y_argmax)):
            if y_argmax[i]!=yhat_argmax[i]:
                
                diff_index.append(i)#실제 label의 위치.
                pred_num.append(yhat_argmax[i]+1)#label 위치에 있는 예측값.
                real_num.append(y_argmax[i]+1)#실제 excel상에 있는 레이블 값.
        print('('+'테스트 사진 수:'+str(output.shape[0])+')',"test_accuracy : ",res,'\n')
        return diff_index,pred_num,real_num
    def Node_box(self,propaType,name,index,data=None):
        if propaType=='f' or propaType=='forward':
            if name=='Relu' or name=='relu':
                dot=np.dot(data,self.weight[index])+self.bias[index]
                activ_dot=self.Relu.forward(dot)
                return activ_dot
            elif name=='sigmoid' or name=='Sigmoid':
                dot=np.dot(data,self.weight[index])+self.bias[index]
                activ_dot=self.sigmoid.forward(dot)
                return activ_dot
            elif name=='tanh' or name=='Tanh':
                dot=np.dot(data,self.weight[index])+self.bias[index]
                activ_dot=self.tanh.forward(dot)
                return activ_dot
        elif propaType=='b' or propaType=='backward':
            if name=='Relu' or name=='relu':
                back=self.Relu.backward(self.backpropagation[index])
                return back
            elif name=='sigmoid' or name=='Sigmoid':
                back=self.sigmoid.backward(self.backpropagation[index])
                return back
            elif name=='tanh' or name=='Tanh':  
                back=self.tanh.backward(self.backpropagation[index])
                return back
    def backward(self):
        self.weight.reverse()
        self.bias.reverse()
        self.propagation.reverse()
        self.node_type.reverse()
        for i,name in enumerate(self.node_type):
            if i==0:
                back=self.Softmax_cross.backward()
                dx=np.dot(back,self.weight[i].T)
                dw=np.dot(self.propagation[i].T,back)
                db=np.sum(back,axis=0)
                self.weight[i]-=self.LR*dw
                self.bias[i]-=self.LR*db
                self.backpropagation.append(dx)
                back=self.Node_box('b',name,i)
                dx=np.dot(back,self.weight[i+1].T)
                dw=np.dot(self.propagation[i+1].T,back)
                db=np.sum(back,axis=0)
                self.weight[i+1]-=self.LR*dw
                self.bias[i+1]-=self.LR*db
                self.backpropagation.append(dx)
            elif i==self.hidden_size_amount:
                back=self.Node_box('b',name,i)
                dw=np.dot(self.batch_x.T,back)
                db=np.sum(back,axis=0)
                self.weight[i+1]-=self.LR*dw
                self.bias[i+1]-=self.LR*db
                self.weight.reverse()
                self.bias.reverse()
                self.node_type.reverse()
                self.propagation.clear()
                self.backpropagation.clear()
            else :
                back=self.Node_box('b',name,i)
                dx=np.dot(back,self.weight[i+1].T)
                dw=np.dot(self.propagation[i+1].T,back)
                db=np.sum(back,axis=0)
                self.weight[i+1]-=self.LR*dw
                self.bias[i+1]-=self.LR*db
                self.backpropagation.append(dx)
    
    


import pandas as pd
import cv2
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

test_data=pd.read_csv("C:/Users/dlrms/OneDrive/Desktop/Mnist/mnist_train.csv")#index_col=0
test_data1=pd.read_csv("C:/Users/dlrms/OneDrive/Desktop/file6.csv",index_col=0)

# 시험 데이터 셋 코딩구역
test_y_data1=test_data1['lable']
test_y_data1=np.array(pd.get_dummies(test_y_data1))
x_test_names=test_data1['name'] # 여기부터 수정시작.
test_x_data1=np.array(test_data1.drop(labels=['lable','name'],axis=1))/255.0#이거lable를 split전에 삭제해버리면 randomstate때문에 lable과 실제 데이터간의 오차가 생기게 됨.
# 끝

# 만 데이터의 자리는 변화하기 때문임.
# 훈련 데이터 셋 코딩구역
test_x_data=test_data.drop(labels=['label'],axis=1)# 이거하면 아래에서 split할 때, 실제 데이터와 lable간의 다름이 발생하게 됨. lable은 여기서 떨어졌지
test_y_data=test_data['label'] # 
x_train,x_test,y_train,y_test=train_test_split(test_x_data,test_y_data,test_size=0.1,random_state=50)
# 끝

#x_test=x_test.drop(labels=["label",'name'],axis=1)#예시로, 아래 사진을 출력할 때 분명 사진은 출력되는 값과 같음. 왜냐하면, y데이터는 lable이 사라지기전에

#split으로 같이 분배가 되었기 때문임.
x_train=np.array(x_train/255.0)
x_test=np.array(x_test/255.0)
y_train=pd.get_dummies(data=y_train)
y_train=np.array(y_train)
y_test=pd.get_dummies(data=y_test)
y_test=np.array(y_test)
NeuralNet= Main(x_train,y_train,[3000,2000,1000],['relu','tanh','relu'])
epoch=1
flag=0
batch=2000 
for epochs in range(0,epoch+1):
    for batchs in range(0,len(x_train),batch):
        forward=NeuralNet.forward(x_train[batchs:batchs+batch],start=batchs,end=batchs+batch,batch=batch)
        back=NeuralNet.backward()
    if epochs==0 or epochs==1 or epochs%10==0:
        print("epoch : ", epochs)
        print("accuracy : ", forward)
diff_index,pred_num,real_num=NeuralNet.test_forward(test_x_data1,test_y_data1)
print("(테스트 실패 사진 수: "+str(len(diff_index))+')','fail_percent',len(diff_index)/len(x_test))
print("모델 예측 숫자 :"+str(pred_num))
print("실제 숫자      :"+str(real_num))
# for i,index in enumerate(diff_index):#index는 예측실패한 yhat_data의 자리.
#     image=np.reshape(x_test[index]*255.0,(28,28)).astype('uint8')
#     image=cv2.resize(image,(128,128))
#     print("name:"+x_test_names.iloc[index],'\t',index)
#     cv2.imshow("",image)
#     cv2.waitKey()