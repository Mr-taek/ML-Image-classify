2. np.argmax(array) : 2d던 1d던 상관없이 1d 취급해서 3,1에 있는 원소를 7로 표현함. 0.0에서 3.1까지는 7번이라서.
1. np.reval(list array-like) : 2d 면 1d로 바꿔줌. y_hat으로 만들때 2번과 시너지가 좋음.
3. np.random.random(1) or ((2,2)) : 0~1사이에 float 값을 출력시켜줌. (1)은 딱 하나[0.2], 2,2 는 2x2 행렬로 0~1사이 값을 만듦.[[~],[~]]
6. np.random.rand(~,~) : 정규분포에서 생성되는 값.
4. a=np.array(?), a.T - > 전치행렬. 반드시 numpy type
5. np.sum(np.array-like,axis=0/1) # numpy array를 더함.
    - axis
        - None : 걍 모든 원소 다 더함
        - 1 : 행기준으로 다 더함
        - 0 : 열 기준으로 다 더함(머신러닝에서 자주 사용하게 됨.)


### list
list.reverse() : 1d이면 끝에 있는 원소를 제일 앞으로, 제일 앞에 잇는 원소를 제일 뒤로.
- 의문점 :
    1. 왜 28,31 번 줄에서의 append는 리스트 하나를 더 포장해서 확장시키는건가? 리스트의 깊이가 3차원[][][]이고 main함수에서는 리스트의 깊이가 2차원 [][].
    ```
    def __init__(self,Real_x,Real_y,hidden_size=3,total_layer=4):
        self.weight=[]
        self.each_layer_Dotproducted_values=[]
        self.layers=total_layer
        self.x=Real_x
        self.y=Real_y
        self.hidden_size=hidden_size
        for times in range(self.layers):
            if times==0:
                self.weight.append(np.random.rand(self.x.shape[1],self.hidden_size))
                self.each_layer_Dotproducted_values.append(np.dot(self.x,self.weight))
            else:   
                self.weight.append(np.random.rand(self.hidden_size,self.hidden_size))
                self.each_layer_Dotproducted_values.append(np.dot(self.each_layer_Dotproducted_values[times-1],self.weight[times]))
    def main(self):
        for times in range(self.layers):
            if times==0:    
                temp=np.dot(self.x,self.weight[times])
                temp=self.Activation(temp)
                self.each_layer_Dotproducted_values.append(temp)
            else:
                print('pro :',self.each_layer_Dotproducted_values,'\n','we :',self.weight,'\n')
                temp=np.dot(self.each_layer_Dotproducted_values[times-1],self.weight[times])
                temp=self.Activation(temp)
                self.each_layer_Dotproducted_values.append(temp)
            print('\n')
    ```
### numpy < - > pandas Relationship
1. 판다스 dataframe은 np 배열과 연산이 가능하다.
2. 판다스의 객체는 넘파이의 연산함수와 사용됐을 때 오류가 난다.
### error
1. unsupported operand type(s) for +: 'NoneType' and 'int'
    1.  NoneType : 변수 안에 암것도 없으면 뜨는 에러.
2. RuntimeWarning: divide by zero encountered in log return -(np.sum(y*np.log(y_hat)))
    1. log에 0이 들어가면 inf를 리턴해서 생기는 문제. log안에 아주 작은 값이라도 넣어야 에러가 안남.
3. The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    1. None이 대입된 변수와 실수 변수와 비교연산자를 사용했을 때 나타남.
    2. numpy 객체와 논리연산을 할 때 만들어짐
        - 경우
            1. np array([0.5,0.25])<0.25 : 이 경우 넘파이 배열안에는 2개의 값이 존재, 0.25와 비교시 에러가 발생함.
                - return [False,False] 
        - 해결책: 파이썬 함수 .any() 와 .all() 을 사용한다
            1. .any() : or와 같은의미, boolean 넘파이 배열 안에서 모든 값을 비교해서 하나라도 True가 있으면 True를 리턴
            2. .all() : and와 같은 의미, 모든 값이 True면 True을 리턴.
4. axis 1 is out of bounds for array of dimension 1
    - 경우
        1. x=(?,2)일 때 x/np.sum(x,axis=1)이면 오류가 남. 단 (?,3)부터는 왼쪽처럼 해도 오류가 안 남.
        2. x=(1,3)일 때 x/np.sum(x,axis=1)이면 오류가 남. 이번에는 axis=1때문임. 행이 하나라서 axis는 0으로 해야 하나봄 시발