
import numpy as np
x=np.array([[21,25.0,3],[32,2,4],[12,5,32]])
n=[]
for i in range(len(x)):
    arg=np.argmax(x[i])
    x[i]*=0
    x[i][arg]=1
print(x)# 소수점이 나타난 원핫코드 나타남. 일단 합격.
if x[0][1]==1:
    print('실수,정수 1과 1.0 비교 가능') # 정상 작동한다. 합격