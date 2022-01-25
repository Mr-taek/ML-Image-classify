import numpy as np

class Network:
    def __init__(self,realx,realy):
        self.x=realx
        self.y=realy
        self.LR=0.005
        self.weight=np.random.randn(self.x.shape[1],self.y.shape[1])*np.sqrt((2.0/self.y.shape[1]))
        self.bias=np.zeros(self.y.shape[1])
        
    def main(self):
        temp=np.dot(self.x,self.weight)+self.bias
        temp=self.relu(temp)
        temp=self.softmax(temp)
        self.update(temp)
        print(np.argmax(temp,axis=1),np.argmax(self.y,axis=1))
    def sigmoid(self,input):
        return 1/(1+np.exp(-input))
    def relu(self,input):
        mask=(input<0)
        out=input.copy()
        out[input<0]*=0.01
        return out
    def softmax(self,input):
        input=input.T
        max=np.max(input,axis=0)
        input=input-max
        soft=np.exp(input)/np.sum(np.exp(input),axis=0)
        return soft.T
    def update(self,soft):
        self.weight-=self.LR*np.dot(self.x.T,soft-self.y)
        self.bias-=self.LR*(np.sum(soft,axis=0))

data=[[ 0.23287639,-1.14992863,1.11334981],
       [ 0.47132952,  0.62612609, 0.64321696],
       [ 0.5430421 ,  1.64517353,  0.13319793],
        [-0.18701142,  0.81181332, -0.26807986],
       [ 0.80608845,  2.16077888, -1.64444971],
       [-0.80636662,  0.00277398,  1.81824277]]
data_answer=[[0,0,1],
             [0,0,1],
             [1,0,0],
             [0,0,1],
             [0,1,0],
             [0,0,1],]

