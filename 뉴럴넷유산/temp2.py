# 활성화 함수에다가 forward랑 백워드 다 넣고.. 
# Relu를 딱 하나만 쓸 것이기 때문에 mask는 list로 안 할 것임.
# sig - > relu - > sofrmax_crossentropy
import numpy as np 
class Sigmoid():
    def __init__(self):
        self.out=None
    def forward(self,input):
        self.out=1/(1+np.exp(-input))
        return self.out
    def backward(self,backpropagation):
        back=backpropagation*self.out*(1-self.out)
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
        input=input.T# broadcast를 위해선 열의 개수가 같아야함.
        max=np.max(input,axis=0)
        self.input=input-max
        self.y_hat=np.exp(self.input)/np.sum(np.exp(self.input),axis=0)# 여기 self.input 실수함.위에서 np exp를 안해서 0으로 나눌때도 있고 오류가남.
        self.y_hat=self.y_hat.T
        print('accuracy : ',self.accuracy(self.y_hat))
        self.loss=self.cross_entropy(self.y_hat)
        return self.loss
    def accuracy(self,y_hat):
        maxy=np.argmax(self.y[self.start:self.end],axis=1)
        maxy_hat=np.argmax(y_hat,axis=1)
        return np.sum(maxy==maxy_hat)/maxy.shape[0]
    def cross_entropy(self,input):
        cross=np.sum(self.y[self.start:self.end]*np.log(input+1e-7),axis=1)#0은 어차피 소멸.. 1이랑 곱해진 log만 살아남고
        entropy=-np.sum(cross)/self.batch
        print(entropy)
        return entropy
    def backward(self,backpropagation=1.0):
        #batch=self.y.shape[0]
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
class Network():
    def __init__(self,Realx,Realy,hidden_size_list,LR=0.01):
        self.x=Realx
        self.y=Realy
        self.LR=LR
        self.hidden_size_amount=len(hidden_size_list)#총 레이어의 숫자
        self.hidden_weight_input_size=hidden_size_list
        self.propagation=[]
        self.backpropagation=[]
        self.weight=[]
        self.bias=[]
        for i in range(self.hidden_size_amount):# 만약 마지막 층값을 추가하고 싶다면 +1을 해야함. 마지막 15가 무시 됨.
            if i==0:
                self.weight.append(np.random.randn(self.x.shape[1],self.hidden_weight_input_size[i])*(1.0/np.sqrt(self.hidden_weight_input_size[i])))
                self.bias.append(np.zeros((self.hidden_weight_input_size[i])))
            elif i==self.hidden_size_amount-1:
                self.weight.append(np.random.randn(self.hidden_weight_input_size[i-1],self.y.shape[1]))
                self.bias.append(np.zeros(self.y.shape[1]))
            else:
                self.weight.append(np.random.randn(self.hidden_weight_input_size[i-1],self.hidden_weight_input_size[i])*(np.sqrt(2.0/self.hidden_weight_input_size[i])))
                self.bias.append(np.zeros((self.hidden_weight_input_size[i])))
        self.sigmoid=Sigmoid()
        self.Relu=Relu()
        self.Softmax_cross=Softmax_crossEntropy(self.x,self.y)
    def forward(self,batch_x,start,end,batch):# ['act':relu,'act':sig].. 근데 이렇게 하면 탠서플로우랑 무슨 차이임?
        self.batch_x=batch_x
        for i in range(self.hidden_size_amount):
            if i==0:# weight[i][start:end]오류, 애초에 weight는 특성에 맞게 되어 있어서 [start:end]는 의미가 없고 오히려 오류 요소.
                #self.bias[i][start:end]도 마찬가지.
                dot=np.dot(batch_x,self.weight[i])+self.bias[i]#[i]제거,batch_x[start:end]가 아니라 그냥 batch_x
                activ_dot=self.sigmoid.forward(dot)
                self.propagation.append(activ_dot)
            elif i==self.hidden_size_amount-1:
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                #dot=self.Relu.forward(dot)# 여기 수정(추가)함.log함수는 음수를 취급하지 않기 때문에 corss사용을 위해 0으로 만든 것임.
                activ_dot=self.Softmax_cross.forward(dot,start,end,batch)#여기 새로 추가함, 손실함수에선 y의 행열이 정확히 필요함.
                #그런데 제거한 결과, 정확도가 평균 80%이상으로 증가, 있을 때는 40%이하이었음. backward도 손봐야함.
                return activ_dot#softmax_cross.backward 안에서 알아서 softmax를 거친 값을 저장함.
            else:
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.Relu.forward(dot)
                self.propagation.append(activ_dot)
    def softmax_accuracy(self,output,y_data):
        output=output.T
        max=np.max(output,axis=0)
        output=np.exp(output-max)
        output=(output/np.sum(output,axis=0)).T
        yhat_argmax=np.argmax(output,axis=1)
        y_argmax=np.argmax(y_data,axis=1)
        res=np.sum(yhat_argmax==y_argmax)/y_data.shape[0]
        print("test_accuracy : ",res,'\n')
    def test_forward(self,x_data,y_data):
        for i in range(self.hidden_size_amount):
            if i==0:
                dot=np.dot(x_data,self.weight[i])+self.bias[i]
                active_dot=self.sigmoid.forward(dot)
                self.propagation.append(active_dot)
            elif i==self.hidden_size_amount-1:
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                #dot=self.Relu.forward(dot)# 여기 수정(추가)함.log함수는 음수를 취급하지 않기 때문에 corss사용을 위해 0으로 만든 것임.
                #그런데 제거한 결과, 정확도가 평균 80%이상으로 증가, 있을 때는 40%이하이었음. backward도 손봐야함.
                self.softmax_accuracy(dot,y_data)
            else:
                dot=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                activ_dot=self.Relu.forward(dot)
                self.propagation.append(activ_dot)
    def backward(self):
        self.weight.reverse()
        self.bias.reverse()
        self.propagation.reverse()
        for i in range(self.hidden_size_amount):#인덱스 순서 확인하자.
            if i==0:#-2의 의미 분해 : -1 은 0을 포함하기 때문이고 또다른 -1은 이전 작품과 달리 y_hat을 다음 노드에 넘겨 주지 않고 인덱스도 0부터 시작임.
                #따라서 모든 propagation의 결과는 해당 노드에 담겨져 있음. 그래서 총 -2를 한 것.
                # 수정 # 다시 생각해보니 propagation은 reverse를 시켰기 떄문에 위는 무의미함. 그래서 다시 i로 바꿈. 이때부터는 i만 넣어주면 됨. 마지막 노드
                #에는 append가 안 붙어서 자동으로 이전 항의 것을 선택하기 때문.
                back=self.Softmax_cross.backward()
                #back=self.Relu.backward(back)
                dx=np.dot(back,self.weight[i].T)
                dw=np.dot(self.propagation[i].T,back)#np dot과 np sum을 아예 layer class의 backward로 해도 될 듯.
                db=np.sum(back,axis=0)
                self.weight[i]-=self.LR*dw
                self.bias[i]-=self.LR*db
                self.backpropagation.append(dx)#이렇게 되면.. backpro는 0,1,2..로 가서 반대가 되긴 할탠데.. 일단 정석대로 가보자. 덮어씌우지 말고.
            elif i==self.hidden_size_amount-1:
                back=self.sigmoid.backward(self.backpropagation[i-1])# 모든 back은 역순으로 인덱스가 흐르는 것이 아니기 때문에 이전 노드에 정보가 있음.
                dw=np.dot(self.batch_x.T,back)
                db=np.sum(back,axis=0)
                self.weight[i]-=self.LR*dw
                self.bias[i]-=self.LR*db
                self.weight.reverse()
                self.bias.reverse()
                self.propagation.clear()
                self.backpropagation.clear()
            else:
                back=self.Relu.backward(self.backpropagation[i-1])
                dx=np.dot(back,self.weight[i].T)
                dw=np.dot(self.propagation[i].T,back)
                db=np.sum(back,axis=0)
                self.weight[i]-=self.LR*dw
                self.bias[i]-=self.LR*db
                self.backpropagation.append(dx)


import pandas as pd
from sklearn.model_selection import train_test_split
# data=pd.read_csv('C:/Users/dlrms/OneDrive/Desktop/Mnist/mnist_train.csv')
# x=data.drop(labels=['label'],axis=1)
# y=data['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)
# y_train=pd.Series(y_train)
# y_train=pd.get_dummies(data=y_train)
# x_train=np.array(x_train/255.0)
# x_test=np.array(x_test/255.0) #이것도 틀렸지만
# y_train=np.array(y_train)
# y_test=pd.get_dummies(y_test)# 이것 안했고
# y_test=np.array(y_test)# np array를 안 해서 오류가 남.
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
# x_train=np.array(x_train)
y_train=pd.get_dummies(y_train)
y_test=pd.get_dummies(y_test)
# y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)

epoch=5
batch=len(x_train)# 들어가는 데이터가 큰데, 배치가 너무 작으면.. 이것도 학습의 결과가 늦어진다..?
first= Network(np.array(x_train),np.array(y_train),[1000,450,100,50,5])#여기 노드를 존나 바꾸면 뭐가 달라지긴 한다. 근데.. 문제는 시발.. 존나 왔다갔다 함.
# 한 레이어에 노드가 존나 많아도 문제가 되는 듯함. layer을 늘리면 어떻게 되는 지 궁굼함.
# 레이어를 늘리니까 더 이상해짐. 아무래도 레이어를 조금 잘 다듬어야 하나봄. 정해진 레이어에만 집중하다보니 레이어를 늘리면 결과가 거 엉성해지는 결과가 등장.
# for i in range(100):# 후우 문제가 .. 이제는 학습이 되는 것 같긴한데? 시발.. 돌릴때마다 자꾸 달라져 ~ 열받네 시발...
#     kbs=first.forward()
#     back=first.backward()

#보통은 배치를 사용하니까.. 이걸로 하자.
for i in range(1000):# 후우 문제가 .. 이제는 학습이 되는 것 같긴한데? 시발.. 돌릴때마다 자꾸 달라져 ~ 열받네 시발...
    for j in range(0,len(x_train),batch):
        kbs=first.forward(batch_x=x_train[j:batch+j],start=j,end=batch+j,batch=batch)
        back=first.backward()
batch=len(x_test)
for i in range(0,len(x_test),batch):
    print(i,'~',i+batch)
    kbs=first.test_forward(x_test[i:batch+i],y_test[i:batch+i])
    print('\n')