# 오차 역전파법은 수치미분보다 빠르다. 수치미분은 구현은 쉽지만 굉장히 느리다는 단점이 있다. 그러나 버그가 숨어 있기 어려운 반면 오자역전파법은 구현하기
# 복잡해서 종종 실수한대.
# 그래서 오차역전파법이 제대로 실행됐는 지 확인하기위해선 역시 수치미분으로 하이퍼파라미터를 계산해서 두 기울이 최종값을 비교해서 아주 작으면 합격.

# 힌트 1.  각 노드는 자신과 '관련된' 계산 외에는 신경쓸 게 없다. 즉 들어오는 데이터와 가중치, 아웃풋에만 집중이 역전파의 특징. " 국소적 계산 "
# 그래프 노드 그리는 것도 국소적 계산일 수도?
# 각 노드마다 손실함수를 구해야 하나 ? 근데 중간 은닉층에는 정답 레이블이 없어서 안 되는데.. 손실함수가 가능한 곳은 최종층 뿐야.
# 합성함수의 미분은 합성함수를 구성하는 함수의 미분의 곱으로 표현할 수 있다.(말 멋있네.) 연쇄법칙은 함성 함수의 미분에 대한 성질.
# 하나 알아낸 것, 일단 모든 역전파의 시작은 출력층 노드에서 시작하고, 당연히 손실함수에 대한 미분을 진행한다.
# 곱셉 노드에선 
import numpy as np
class softmax_layer:
    def __init__(self):
        self.x=None
        self.w=None
        pass
    def forward(self,x,w):#순전파
        self.x=x
        self.w=w
        out=np.dot(self.x,self.w)
        return out
    def backward(self,dout):
        dx=dout
class sigmoid_layer:
    
    def forward(self,x,w):
        y_hat=np.dot(x,w)
        return y_hat # dout으로 입장예정.
    def main(self):
        a=0
    def backward(self,dout,weight,x,bias):
        dout=self.sigmoid(np.dot(x,weight)+bias)
        weight=weight*dout*x# 현재 층의 가중치 값 바꾸기. x=(891,165) weight_sig=(165,?)
        x=x*dout*weight#다른층에서 들어온 y_hat, 이전 층 애들도 똑같이 해야지.
        bias=dout
    def sigmoid(self):
        a=0
class Relu_layer:
    def Relu(self,input):
        if input>0:
            return input
        else: return 0
    def backward(self,x,weight):
        dout=self.Relu(x)
        weight=weight*
from collections import OrderedDict
layers=OrderedDict()
layers['a1']=[52,123,123]
layers['re']=1/56
layers['a2']=56
print(layers[2])