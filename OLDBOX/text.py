import numpy as np

class flow:
    
    def __init__(self,x,y,w):
        self.x=None
        self.weight1=np.random.random((500,y.shape[1]))
        self.bias1=np.random.rand()
    def predict(self,x,w,b):# 2,3 * 3,2 - 2,2
        y_hat=np.dot(x,w)+b
        #print(y_hat,'\n')
        #y_hat1=self.softmax(y_hat)
        for i in range(y_hat.shape[0]):
            for j in range(y_hat.shape[1]):
                if y_hat[i][j]<0:
                    y_hat[i][j]=0
        y_hat=np.dot(y_hat,self.weight1)+self.bias1
        y_hat2=self.softmax1(y_hat)
        #print(y_hat2)
        return y_hat2
    #def softmax(self,y_hat):
        sum=0
        for i in range(y_hat.shape[0]):
            for j in range(y_hat.shape[1]):
                sum+=np.exp(y_hat[i][j])
            for k in range(y_hat.shape[1]):
                y_hat[i][j]=np.exp(y_hat[i][j])/sum
            sum=0
        return y_hat
    def softmax1(self,y_hat): # 최종 출력에서 y_hat값을 가져오기.
        #y_hat=np.exp(y_hat)/np.sum(np.exp(y_hat))#전체 값을 다 ?/100처럼 만들어 서 비율로 나누기.
        Ny=[]
        for i in range(y_hat.shape[0]):
            Ny.append(np.exp(y_hat[i])/np.sum(np.exp(y_hat[i])))
        Ny=np.array(Ny)
        return Ny
    def loss(self,y_hat,y): # 2,2 2,2
        #print(y_hat,'\n')
        #for i in range(len(y)):
        #    errors.append(((y_hat[i]-y[i])**2))
        error=(y_hat-y)**2
        #print(y,'\n',y_hat,'\n',error,'\n')
        errors=[]
        for i in range(len(error)):
            errors.append(np.sum(error[i]))
        errors=np.array(errors)# 1,2가 됨;
        #errors=np.resize(errors,(,1))
        errors=errors
        #print(errors)
        return errors
    def main(self,x,y,w,bias,iter):
        y_hat=0
        for i in range(iter):
            y_hat=self.predict(x=x,w=w,b=bias)
            loss=self.loss(y_hat,y)#softmax를 거친 y_hat
            #print('error rate : ',loss,'\n')
            w,bias=self.gradient(x,y_hat,y,w,0.01,bias)# reshape weight
        ret=self.accuracy(y_hat,y)
        print(ret,'\n',y)
    def gradient(self,x,y_hat,y,w,lr,bias):
        func=(y_hat-y)
        w-=lr*(np.dot(x.T,func))/x.shape[0]# 요 mean()이 조금.. 거슬리네..numpy함수임.
        bias-=lr*(func)/x.shape[0]
        return w,bias
    def accuracy(self,y_hat,y):
        accuracy=0
        ret=[]
        for i in range(y.shape[0]):
            if y[i].argmax()==y_hat[i].argmax():
                accuracy+=1
                arg=y_hat[i].argmax()
                y_hat[i]*=0
                y_hat[i][arg]=1
                ret.append(y_hat[i])
        print(accuracy/y.shape[0])
        return np.array(ret)
        

xx=np.array([
    [1,2,3,5,6],
    [12,32,5,4,6],
    [56,23,12,2,3],
    [32,41,21,1,5]
])
yy=np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [1,0,0,0]
])
ww=np.array([# 뭐하나만 잘 못되도 can only concatenate list (not "float") to list , 요런거 뜨는데 무서운거는 여기가문제인데 여기 라인이 안 뜸.
    [0.2,-0.89,-0.56,0.78],
    [0.9,0.56,0.35,-0.5],
    [-0.8,0.45,0.65,-0.65],
    [0.52,0.23,2.32,3.65],
    [1.56,2.65,0.36,0.89]
])
#w1=np.array([[ 0.11665587,-0.80665587],
 #[ 0.4795233  , 0.9804767 ],
 #[-0.06660221,-0.28339779]])
#b=-0.7

import seaborn as sns
import pandas as pd
data=sns.load_dataset('titanic')
data['who']=data['who'].astype('category')
data.dropna(axis=0,subset=['age'],inplace=True)
#temp3=data['class'].map(arg={'First':1,'Second':2,'Third':3})
#data=pd.concat([data,temp3],axis=1)
temp=pd.get_dummies(data['class'],prefix='cla')
temp1=pd.get_dummies(data['who'])
aloness=pd.get_dummies(data['alone'],prefix='alone')
#data.drop(labels=['deck','who','class'],axis=1,inplace=True) # 굳이 필요는 없어.
data=pd.concat([data,temp,temp1],axis=1)
x=pd.concat(objs=[data[['fare','age']],temp1,aloness],axis=1)
#x=x.dropna(axis=0)# age에 nan값이 많이 있었음.
y=np.array(temp)

w=np.random.random((x.shape[1],500))
bias=np.random.rand()
obj=flow(x,y,w)

obj.main(x,y,w,bias,100)

t=np.array([[1,1,1],[1,2,3]])
print(t.sum(axis=1).mean())
#[1 0 1 0 0 1]
