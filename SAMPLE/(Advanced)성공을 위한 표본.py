import numpy as np
#xavier 초기 값 : 가중치의 초기값은 치우치지 말고 골고루 분포해야함. 앞 계층의 노드가 n개면 '표준편차'가 1/root(n)인 분포를 사용하면 됨.
#Relu에 특화된 초기값, He 초기값.
#배치 정규화 알고리즘??
class Let:
    def __init__(self,hidden_size):
        self.w=[]
        self.hidden_size=hidden_size
        self.Main_time=0
        for i in range(hidden_size):
            self.w[i]=np.random.randn(~,hidden_size)
            if i ==hidden_size:
                self.w[i]=np.random.rand(~,y.shape[1])
        
    def main(self,x_,y_):#recursive
        #for i in range(self.hidden_size): 오히려 for문의 recursive는 stuck될 것임.
        #Affine
        output=np.dot(x_,self.w[self.Main_time])
        if self.Main_time==self.hidden_size:
            output=self.softmax(output)
            self.Main_time=0
            print(self.accuracy(output))
        else: 
            output=self.Leaky_Relu(output)
            self.Main_time+=1
            self.main(output,y_)
        