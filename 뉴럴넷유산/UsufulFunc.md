1. accuracy(self,x,y):
```
accuracy(self,x,y):
    y_hat=self.predict(x)
    y_hat=np.argmax(y,axis=1) # 1은행. 행에는 각 속성의 해당 속성일 확률 값이 있음. 행에서 가장 큰 값을 가져옴
    if y.ndim != 1 :
        y=np.argmax(y,axis=1) # y도 각 행에 y_hat과 같은 속성이 담겨있으나 진짜 정답이 있음. 따라서 진짜 정답 인덱스만 가져옴
    accuracy(np.sum(y==t))/float(x.shape[0]) # 둘다 (x행사이즈,1)일 것임. 각 행의 원소가 같은 횟수를 모두 세는 거고 맞으면 1씩 증가하기 때문에 
    맞은 것/전체데이터로 계산이 가능.
    return accuracy
``` 

2. softmax
```
def softmax(self,y_hat): # 최종 출력에서 y_hat값을 가져오기.
    y_hat=np.exp(y_hat)/np.sum(np.exp(y_hat))#전체 값을 다 ?/100처럼 만들어 서 비율로 나누기.
    return y_hat
```
```
a=[0.5,2.9,4.5,5.6]# 막 200넘는 숫자 넣으면 소수점 4째자리 이하까지도 나와버려서 요런 소수일 때가 좋음..
b=np.exp(a)/(np.sum(np.exp(a)))
print(b)
```