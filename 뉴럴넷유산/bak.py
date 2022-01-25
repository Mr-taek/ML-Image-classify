#학습 데이터로 forward연산을 통해 loss값을 구함
#각 layer별로 역전파 학습을 위해 중간값을 저장.
# 손실함수를 가중치, 편향 으로 미분하여 연쇄법칙을 이용하여미분, 각 layer를 통과할 때 마다 저장된 값을 이용
# 오류룰 전달하면서 학습마라미터를 조금씩 갱신

##오차 역전파 학습의 특징
# 손실함수를 통한 평가를 한 번만 함.
# 학습 소요시간이 매우 단축, 중간값을 모두 저장하기 때문에 메모리를 많이 사용

## 신경망 학습에 있어서 미분가능의 중요성
# 미분이 가능해야한다.

# mnist 관련 영상은 36분 45초
# 255.0으로 스케일링.
def back_sigmoid(self,y,dout):#y는 1/(1+np.exp(-x))의 결과값
    back=dout*y*(1-y)#dout은 backpropagation
    return back
def sigmoid(self):
    a=0
class relu():
    def __init__(self) -> None:
        self.mask=[]
        self.mask_index=-1
    def relu(self,input):
        mask=(input<=0)
        self.mask.append(mask)
        self.mask_index+=1
        input[mask]=0
    def back_relu(self,back):
        back[self.mask[0]]=0
        #dx=dout  이건 굳이 해야하나?
        return back
import numpy as np
from numpy.core.numeric import cross
class Layer():
    def __init__(self):
        self.w=np.random.randn(3,2)
        self.b=np.random.randn(2)
        self.x=None
        self.dw=None
        self.db=None
        
    def forward(self,x):
        self.x=x
        out=np.dot(x,self.w)+self.b
        return out
    def back(self,back):
        dx=np.dot(back,self.w.T)
        self.dw=np.dot(self.x.T,back)
        self.db=np.sum(back,axis=0)
        return dx
np. random.seed(111)
f=Layer()
x=np.random.rand(2,2)
y=f.forward(x)
print(x,y)

## hyper parameters
epochs=1000
LR=1e-3

def softmax(x):
    if x.ndim==2:
        x=x.T
        x=x-np.max(x,axis=0)
        y=np.exp(x)/np.sum(np.exp(x),axis=0)
        return y.T#그냥 취향인듯.
    x=x-np.sum(x)#1차원일경우.
    return np.exp(x)/np.sum(np.exp(x))
def mean_sq(y_hat,real_y):
    return 0.5*np.sum(y_hat-real_y)**2#0.5는 1/2
def crosss_entropy(y_hat,real_y):
    if y_hat.ndim==1:
        real_y=real_y.reshape(1,real_y.size)
        y_hat=y_hat.reshape(1,y_hat.size)
    if y_hat.size==real_y.size:
        real_y=real_y.argmax(axis=1)
    batch_size=y_hat.shape[0]
    return -np.sum(np.log(y_hat[np.arange(batch_size),real_y]+1e-7))/batch_size
def softmax_loss(x,real_y):
    pre=softmax(x)
    return crosss_entropy(pre,real_y)

class Relu():
    def __init__(self):
        self.out=None
    def forward(self,x):
        self.mask=(x<0)
        out=x.copy()
        out[x<0]=0
        return out
    def backward(self,dout):
        dout[self.mask]=0
        dx=dout
        return dx
class Sigmoid():
    def __init__(self):
        self.out=None
    def forward(self,x):
        out=1/(1+np.exp(-x))
        return out
    def backward(self,dout):
        dx=dout*self.out*(1-self.out)
class Layer():
    def __init__(self,w,b):
        self.w=w
        self.b=b
        self.x=None
        self.ori_x_shape=None
        self.dl_dw=None
        self.dl_db=None
    def forward(self,x):
        self.ori_x_shape=x.shape
        x=x.reshape(x.shape[0],-1)
        self.x=x
        out=np.dot(self.x,self.w)+self.b
        return out
    def backward(self,dout):
        dx=np.dot(dout,self.w.T)
        self.dl_dw=np.dot(self.x.T,dout)
        self.dl_db=np.sum(dout,axis=0)
        dx=dx.reshape(self.ori_x_shape)
        return dx
class softmax():
    def __init__(self):
        self.loss=None
        self.y=None
        self.t=None
    def forward(self,x,t):
        self.t=t
        self.y=softmax(x)
        self.loss=crosss_entropy(self.y,self.t)
        
        return self.loss
    def backward(self,dout=1):
        batch_size=self.t.shape[0]
        if self.t.size==self.y.size:
            dx=(self.y-self.t)/batch_size
        else:
            dx=self.y.copy()
            dx[np.arange(batch_size),self.t]-=1
            dx=dx/batch_size
        return dx

class Mymodel():
    def __init__(self,input_size,hidden_size,output_size,activation='relu'):
        self.input_size=input_size
        self.output_size=output_size
        self.hidden_size=hidden_size
        self.hidden_layer=len(hidden_size)
        self.params={}
        self.__init_weights(activation)
        activation_layer={'sigmoid':Sigmoid,'relu':Relu}
        self.hidden_layers=
# 가중치 he초기화 , 사비에르 초기화.

def accuracy(self,x,true_y):
    pred_y=self.predict(x)
    pred_y=np.argmax(pred_y,axis=1)
    accuracy=np.sum(pred_y==true_y)/float(x.shape[0])
    return accuracy

def gradient(self,x,t):
    self.loss(x,t)
    
    dout=1
    dout=self.last_layer.backward(dout)
    layers=list(self.layers.values())
    layers.reverse()
    for layer in layers:
        dout=layer.backward(dout)
    grad={}
    for idx in range(1,self.hidden_layer_num+2):#초기화할때 끝에 2개더 추가해서
        grads['w'+str(idx)]=self.layers['layer'+str(idx)].dl_dw
        grads['b'+str(idx)]=self.layers['layer'+str(idx)].dl_db
        return grads
    
    
model=Mymodel(28*28,[100,64,32],10)#hiddensize를 100 64 32로 줄이기가 가능..?

train_lost_lost=[]
train_acc_list=[]#코랩에서는 계속 모델이 저장되기 때문에 리스트 안에 정확도를 계속 '저장'이 가능.
for epoch in range(eopchs):
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=X_train[batch_mask]
    y_batch=y_train[batch_mask]
    
    grad=model.gradient(x_batch,y_batch)
    
    for key in model.params.keys():
        model.params[key]-=LR*grad[key]
        
    loss = model.loss(x_batch,y_batch)
    train_lost_lost.append(loss)
    
    if epoch%50==0:
        train_acc=model.accuracy(x_train,y_train)
        test_acc=model.accuracy(x_test,y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("epoch {}")