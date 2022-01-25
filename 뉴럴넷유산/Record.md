- A=data , b=A 일 경우 b는 A와 같은 메모리를 공유해서 b가변하면 A도 변하게 된다. 따라서 b=A.copy()를 사용해서 새로운 메모리를 할당한다.

- (5,) 같은 형상이 나올경우 dot-product가 안 되는 경우가 있는데, 이 때 ! reshape(1,x.size)를 사용한다. x.size은 일종에 len(x)와 같은 기능이며, 1차원 배열을 2차원 배열 형상으로 만들기 위해서 사용된다. x.reshape(1,x.size)의 결과는 1차원형상에서 2차원으로 변환된다. [1,2,3] -> [[1,2,3]]
- Backpropagation : croos - softmax방식으로 할 경우, y_hat-y 의 순서가 중요. 이게 역전파의 첫단추이고 이게 이상하게 되어버리면 가중치가 exponential만큼 증가하게 되어 것잡을 수 없이 커지게 됨.
- np.dot(np.array(self.each_layer_Dotproducted_values[(self.layers-2)-back_times]) , 이 때 오해한 것이, layer_~은 y_hat값을 저장하는 곳이다. back을 할 때 y_hat을 사용하는 것이 아니고 그 이전에 y_hat을 사용해야 한다. 그래서 self.laters-1이 아닌 -2이다.

- randn 함수로 -가 포함된 가우시안 표준정규분포를 생성하면 rand 의 음수가 없는 표준정규분포보다는 출력값이 다양.. 단 여전히 문제.

- sigmoid는 x값이 1.5이상이 되는 순간 의미가 없어진다. 1에 수렴하기 때문이다.

- 가중치의 합이 몇 만을 넘는 것을 막을 수가 없다는 결론에 도달했다. 하지만 분명히 출력되는 값에는 어떤 의미가 담겨 있는 듯 했다. 즉 이 의미를 지켜주면서 -1.0~1.0사이로 만들어 줄 수만 있다면 충분히 할 만하다.

- 1개 데이터가 아닌 bactch사이즈에서의 bias는 np.sum(backpropagation,axis=0)로 하고 업데이트 해준다.

- batch 사이즈를 잘 확인한다, 또한 argmax(axis=1)의 결과는 batch크기만큼의 행의 개수에서, 각 행의 가장 큰 값을 갖는(softed max)인덱스를 반환한다.