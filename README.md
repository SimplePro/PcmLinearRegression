### This is my custom LinearRegression 😎

코드는 다음과 같이 쓸 수 있다.
--------------------

``` python
# 생성
linearRegression = CustomLinearRegression()

# 학습
linearRegression.fit(heights, weights, 10000)

# 예측 값
pred = linearRegression.predict(heights)

# 정확도
accuracy = linearRegression.evaluation(weights, pred)

# 실제 값의 분포와 선형회귀를 그래프로 표현.
linearRegression.evaluation_graph(heights, weights, pred)
```

전체코드
-----------

``` python
class CustomLinearRegression():
    def __init__(self):
        self.a = 0  # x 의 계수
        self.b = 0  # y 절편
        self.x_average = 0  # x 값 평균
        self.y_average = 0  # y 값 평균
        self.x_dispersion = 0  # x 값 분산
        self.y_dispersion = 0  # y 값 분산
        self.r = 0  # 상관계수
        
    # 학습
    def fit(self, X=None, y=None, d_len=None):
        if X is None or y is None:
            raise Exception("x and y cannot be None.")
        if d_len is None:
            raise Exception("shape cannot be None.")
        self.fit_logic(X, y, d_len)
            
    # 학습 로직
    def fit_logic(self, X, y, d_len):
        ## 평균 구하기
        self.x_average = sum(X) / d_len
        self.y_average = sum(y) / d_len
        
        ## 표준편차 구하기
        for i in X:
            self.x_dispersion += (i - self.x_average)** 2
        self.x_dispersion = np.sqrt(self.x_dispersion / d_len)
        
        for i in y:
            self.y_dispersion += (i - self.y_average)** 2
        self.y_dispersion = np.sqrt(self.y_dispersion / d_len)
    
        ## 상관계수 구하기
        r = 0
        for x_d, y_d in zip(X, y):
            r += ((x_d - self.x_average) / self.x_dispersion) * ((y_d - self.y_average) / self.y_dispersion)
        r = r / d_len - 1
        
        ## 기울기
        self.a = r * self.y_dispersion / self.x_dispersion
        
        ## 절편
        self.b = abs(self.a * self.x_average - self.y_average)
    
    # 예측
    def predict(self, x):
        pred = np.array([])

        for i in x:
            pred = np.append(pred, [b1 * i + b0])
        pred *= 0.1
        return pred
    
    # 평가
    def evaluation(self, y, pred):
        accuracy_list = []
        for r, p in zip(y, pred):
            accuracy_list.append(1 - abs(r - p))
        accuracy = abs(sum(accuracy_list) / len(accuracy_list))
        return accuracy
    
    # 선형회귀와 데이터의 그래프를 그려주는 함수.
    def evaluation_graph(self, X, y, pred, sca_col="red", pre_col="blue"):
        try:
            plt.scatter(heights, weights, color=sca_col)
            plt.plot(heights, y_pred, color=pre_col)
        except:
            raise Exception("import matplotlib.pyplot as plt")
```

