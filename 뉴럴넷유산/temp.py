import numpy as np
from numpy.lib.histograms import histogram

class Network:
    def __init__(self,x,y,total_layer=2,hidden=50):
        self.weight=[]
        self.bias=[]
        self.propagation=[]
        self.backpropagation=[]
        self.mask=[]
        self.LR=0.01
        self.mask_index=0
        self.layers=total_layer-1
        self.x=x
        self.y=y
        for i in range(self.layers+1):
            if i==0:
                self.weight.append(np.random.randn(self.x.shape[1],hidden))
                self.bias.append(np.zeros(hidden))
            elif i==self.layers:
                self.weight.append(np.random.randn(hidden,self.y.shape[1]))
                self.bias.append(np.zeros(self.y.shape[1]))
            else:
                self.weight.append(np.random.randn(hidden,hidden))
                self.bias.append(np.zeros(hidden))
        #print(self.x.shape)
    def forward(self):
        times=0
        for i in range(self.layers+1):#forward
            if i==0:
                temp=np.dot(self.x,self.weight[i])+self.bias[i]# i=0
                temp=self.forward_relu(temp)
                self.propagation.append(temp)
            elif i==self.layers:
                temp=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                temp=self.softmax(temp)
                #temp=self.cross_entropy() #얘는 일단 내일 하자. 책을 참고해야겠어. 
                self.propagation.append(temp)
                accuracy=self.accuracy(temp)
                #print(self.weight[0:5])
                print('accuracy : ',accuracy,'\n')
            else:
                temp=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
                temp=self.forward_sigmoid(temp)
                self.propagation.append(temp)
            # else:
            #     temp=np.dot(self.propagation[i-1],self.weight[i])+self.bias[i]
            #     temp=self.forward_relu(temp)
            #     self.propagation.append(temp)
            times=i
        self.backpropagation=self.propagation
        for j in range(times,-1,-1):#backward, times는 그대로 나오는데 -1을 끝으로 해야함. 1에서 -1하면 0이고 0=0이라 멈추는 듯.
            if j==self.layers:
                self.backpropagation[j]=(self.backward_soft_cross(self.propagation[j]))#y_hat->cross->softmax
                self.weight[j]=self.weight[j]-self.LR*np.dot(self.propagation[j-1].T,self.backpropagation[j])
                self.bias[j]=self.bias[j]-self.LR*np.sum(self.backpropagation[j],axis=0)
                self.backpropagation[j-1]=np.dot(self.backpropagation[j],np.array(self.weight[j]).T)#다음층으로 역전파 값 넘기기
            elif j==0:
                self.backpropagation[j]=self.backward_relu(self.backpropagation[j])
                self.weight[j]=self.weight[j]-self.LR*np.dot(self.x.T,self.backpropagation[j])#마지막은 항상 0-1=-1.. x를 넣어야함.
                self.bias[j]=self.bias[j]-self.LR*np.sum(self.backpropagation[j],axis=0)
            else:
                self.backpropagation[j]=self.backward_sigmoid(self.backpropagation[j])
                self.weight[j]=self.weight[j]-self.LR*np.dot(self.propagation[j-1].T,self.backpropagation[j])
                self.bias[j]=self.bias[j]-self.LR*np.sum(self.backpropagation[j],axis=0)
                self.backpropagation[j-1]=np.dot(self.backpropagation[j],np.array(self.weight[j]).T)
            # else:
            #     self.backpropagation[j]=self.backward_relu(self.backpropagation[j])#받은 역전파 가공
            #     self.weight[j]=self.weight[j]-self.LR*np.dot(self.propagation[j-1].T,self.backpropagation[j])
            #     self.bias[j]=self.bias[j]-self.LR*np.sum(self.backpropagation[j],axis=0)
            #     self.backpropagation[j-1]=np.dot(self.backpropagation[j],np.array(self.weight[j]).T)
    def forward_sigmoid(self,input):
        return 1/(1+np.exp(-input))
    def forward_relu(self,input):
        mask=(input<=0)
        self.mask.append(mask)
        self.mask_index+=1
        input[mask]=0
        return input
    def backward_relu(self,backpropagation):#다 하기전에.. 이녀석 성능을 검사해봐야함. 제대로 돌아가는 지 직접 봐야겠음.
        backpropagation[self.mask[self.mask_index-1]]=0#인덱스가 마지막까지 1이 더해져서 하나를 빼고 해야함.
        self.mask_index-=1
        return backpropagation
    def forward_sigmoid(self,input):#soft-cross 는 sigmoid한 상태에서 쓰인다고 해서..
        a=0
    def backward_sigmoid(self,backpropagation):
        return backpropagation*(1-backpropagation)
    def backward_soft_cross(self,backpropagation):
        temp=(backpropagation-self.y)/self.y.shape[0]
        return temp
    def softmax(self,input):
        for i in range(len(input)):
            input[i]=np.exp(input[i])
            summation=np.sum(input[i])#axis 불필요, 어차피 한 행이라서
            input[i]=input[i]/summation
        return input
    # def crsoo_entropy(self,input):
    #     for i in range(len(input)):
            
    def accuracy(self,output):
        output=np.argmax(output,axis=1)
        y=self.y
        y=np.argmax(self.y,axis=1)
        print(output,y)
        accuracy=np.sum(output==self.y)/len(output)
        return accuracy
    
data=np.array([[ 0.23287639,-1.14992863,1.11334981],
       [ 0.47132952,  0.62612609, 0.64321696],
       [ 0.5430421 ,  1.64517353,  0.13319793],
       [-0.18701142,  0.81181332, -0.26807986],
       [ 0.80608845,  2.16077888, -1.64444971],
       [-0.80636662,  0.00277398,  1.81824277],
       [-0.75730109,  0.28386431,  0.08322091],
       [-0.70577269, -1.33762013,  1.31687382],
       [-0.02757098,  0.69070475, -2.78592177],
       [-0.07385293, -1.23482431,  0.96617665],
       [ 2.05162208, -1.15833745,  0.29309905],
       [ 0.01252556,  0.09500633, -0.03724772]])
data_answer=[[1,0,0,0],
             [0,1,0,0],
             [1,0,0,0],
             [0,0,0,1],
             [0,1,0,0],
             [0,0,1,0],
             [1,0,0,0],
             [0,0,0,1],
             [0,0,1,0],
             [1,0,0,0],
             [0,1,0,0],
             [0,0,0,1]]

first=Network(data,np.array(data_answer))
x=np.array([[1,2,3],[1,2,3]])
print(x.reshape(1,-1))
# for i in range(100):
#     first.forward()