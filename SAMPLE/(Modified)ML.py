import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
# 심각한  문제. 2시간 소요. x,y둘다 타입이 다를경우 오류발생. pa 와 np는 서로 사용해도 결국 오류가 난다.
# 결국에는 손실함수를 사용하려면 -1~1 사이 값이여야 적당. 또는 
start1=time.time()

# 함수 만들어서 forward인지 backward인지를 구분해주자..
class input:
    def __init__(self,True_x=None,True_y=None):#output_size 제거
        self.original_input=0#backpropagation용, 미분용
        self.original_y=True_y
        self.w=np.random.randn(True_x.shape[1],True_y.shape[1])#아 init이 제일 먼저 실행돼서 None값을 먼저 인식하지.
        self.b=np.random.random(True_y.shape[1])
        self.dw=None
        self.db=None
    def forward(self,x_input,y_input):
        output=np.dot(x_input,self.w)+self.b
        #생각해보니 .. 경사하강법을 쓰려면 출력값이 0~1사이어야 함. 활성화 함수를 돌려야 해.Relu쓰자. 생각해보니 input 데이터가 1이하라면 경사하강법 쌉 가능이지.
        output=self.softmax(output)
        print(output[:15])
        #경사하강법이 아닌 cross가 loss임으로.. + 업데이트 동시진행.
        #temp=(y_input*(1/output))#여기도 존나 오래 걸릴 것 같은데.. 1/0.06이렇게 되거나 .. 1/ 0.5이렇게 되자녀.
        temp=(output-y_input)# 와 시발? 여기가 문제인가? 아니 여기가 문제는 맞어.
        self.w=self.w-0.01*np.dot(x_input.T,temp)# - * -(0.01~) 이라 +
        self.b=self.b-0.01*np.sum(temp,axis=0)
        print(self.accuracy(output,y_input),'\n')
        return output
    def softmax(self,input):
        input=np.exp(input)
        for i in range(input.shape[0]):
            sum=np.sum(input[i])# 각 행마다 총 합을 .. 여기서 꽤 느려지지 않을까..
            input[i]=input[i]/sum
        return input
    def sig(self,output):
        output=1/(1+np.exp(-output))
        return output
    def backward(self,input):
        #Leaky_Lelu derivative
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                if input[i][j]<=0:
                    input[i][j]*=0.1
        #Affine derivative
        self.dw=np.dot(self.original_input.T,input)
        self.db=np.sum(input,axis=0)# 열기준으로 싹 다 더함.
        self.w=self.w-0.01*self.dw
        self.b=self.b-0.01*self.db
    
    def Leaky_Relu(self,input):
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                if input[i][j]<=0:
                    input[i][j]*=0.1
        return input
    def accuracy(self,y_hat,y):
        y=np.argmax(y,axis=1)
        y_hat=np.argmax(y_hat,axis=1)
        #print(y[:30],'\n',y_hat[:30])
        return np.sum(y==y_hat)/len(y)     
    def cross_entropy(self,output,True_y):
        #print(np.log(output)[:5],'\n',output[:5])
        output=-1*np.sum(True_y*np.log(output),axis=1)
        output=np.sum(output,axis=0)/len(output)
        return output
class hidden:
    def __init__(self,True_y):#,feature_units=0 삭제.
        self.input=None#backpropagation용
        self.y=True_y
        self.w=np.random.randn(self.input.shape[1],self.y.shape[1])#self.input은 self.w보다 먼저 초기화 되기에 가능.
        self.b=np.random.random(self.y.shape[1])
        self.dw=None
        self.db=None
    def forward(self,input):
        self.input=input#생각해보니 원래의 input은 보존시켜야 함.
        input=np.dot(input,self.w)+self.b# 만약 self.input이 l value면 원래의 값이 보존되지가 않음.
        input=self.softmax(input)
        #cross_entropy
        loss=self.cross_entropy(input)
        print("loss : ",loss)
        
    def softmax(self,input):
        input=np.exp(input)
        for i in range(input.shape[0]):
            sum=np.sum(input[i])# 각 행마다 총 합을 .. 여기서 꽤 느려지지 않을까..
            input[i]=input[i]/sum
        return input
    def cross_entropy(self,input):
        print(input)
        output=-1*np.sum(self.y*np.log(input),axis=1)#여기도 시간이 꽤 걸리지 않을까 싶은데. (~,1)을 아웃.
        return output

temp=np.array([
    [1,2,3,4,5],
    [2,3,4,5,6],
    [6,7,2,1,3],
    [5,2,1,7,8]
],dtype=float)#새로운 정보지만.. 데이터 타입을 명시해야함. 어이가 없지만 자동 형변환이 일어나지 않음. 아니면 list안에 하나라도 float값이 있어야함.
y=[
    [0,1,0],
    [1,0,0],
    [0,1,0],
    [0,0,1]
]

data=pd.read_csv('C:/Users/dlrms/OneDrive/Desktop/archive (1)/mnist_train.csv')
x=data.drop(labels=['label'],axis=1)
y=data['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)
x_train=np.array(x_train)/255.0
y_train=pd.Series(y_train)
y_train=pd.get_dummies(data=y_train)
y_train=np.array(y_train)
batch_size=1000
obj1=input(x_train,y_train)# 나중에 소멸자써서 값리턴하고 다음걸로. 그럼 차라리 소멸자에다가 for문을 쓴다면?
for _ in range(4):
    for i in range(0,x_train.shape[0],batch_size):# epoch , 0~59999 총 6만 개.
        x=x_train[i:i+batch_size]
        y=y_train[i:i+batch_size]
        obj1.forward(x,y)


