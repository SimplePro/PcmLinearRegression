### This is my custom LinearRegression 😎

설명
------------
1. 먼저 학습을 시작하기 전에 데이터들을 전처리한다. (preprocessing)
- x 값이 비슷한 y 값들끼리의 평균값을 구한다.  
``` python
self.data = list(zip(X, y))
self.data = np.sort(self.data)
self.data = pd.DataFrame(self.data, columns=["X", "y"])

def duplicate(x):
    duplicate_data = self.data[(self.data["X"] > (x - 0.1)) & (self.data["X"] < (x + 0.1))]["y"]
    return sum(duplicate_data) / len(duplicate_data)

self.data["y"] = self.data["X"].apply(lambda x: duplicate(x))
self.data = self.data.drop_duplicates(["y"], keep="first")
self.data = self.data.reset_index(drop=True)
```

2. 전처리한 데이터에서 up_rate 을 기준으로 증가량을 측정하여 증가량의 평균을 구한다.
- 증가량으로 기울기를 구한다.
``` python
for i in range(self.data.shape[0]):
    try:
        up.append((self.data.iloc[i, 1] - self.data.iloc[i + self.uprate, 1]) / (
                self.data.iloc[i, 0] - self.data.iloc[i + self.uprate, 0]))
    except:
        pass

for i in reversed(range(self.data.shape[0])):
    try:
        up.append((self.data.iloc[i, 1] - self.data.iloc[i - self.uprate, 1]) / (
                self.data.iloc[i, 0] - self.data.iloc[i - self.uprate, 0]))
    except:
        pass

# 기울기 구하기
self.a = sum(up) / len(up)
```


3. 절편은 첫번째 x 값에 대한 예측값 - 첫번째 x 값에 대한 y 값 을 하여 구할 수 있다.  
``` python
# 절편 구하기
self.b = self.data.iloc[0, 1] - (self.a * self.data.iloc[0, 0])
```

그렇게 y = ax + b 형태의 일차함수(LinearRegression) 그래프를 얻을 수 있다.
``` python
# 정보
def info(self, ifpr=True):
    if ifpr:
        print(f"y = {self.a}x + ({self.b})")
    return self.a, self.b
```

4. X 값과 y 값으로 평가 그래프를 그릴 수 있다.
``` python
def evaluation_graph(self, X, y):
    plt.scatter(X, y, label="original")
    plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], color="red", label="preprocessing")

    pred = self.predict(self.data.iloc[:, 0])
    plt.plot(self.data.iloc[:, 0], pred, color="yellow", label="predict")

    plt.legend()

    plt.show()
```

<div>
<img width="285" alt="evaluation_graph_img" src="https://user-images.githubusercontent.com/66504341/103880075-ff0de680-511b-11eb-8d5c-d9ba8cf9c559.PNG">
</div>

5. 예측은 predict 메소드를 이용하여 할 수 있다.
``` python
# 예측
def predict(self, X):
    y_pred = []
    for i in X:
        y_pred.append(self.a * i + self.b)
    return y_pred
```

  
다음과 같이 사용할 수 있다
--------------------

``` python
# 임포트
from CustomML.CustomRegression import CustomLinearRegression

# 생성
lr_model = CustomLinearRegression()

# 학습
lr_model.fit(heights, weights)

# 실제 값의 분포와 선형회귀를 그래프로 표현.
lr_model.evaluation_graph(heights, weights)
```

전체코드
-----------

``` python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class CustomLinearRegression():
    def __init__(self, up_rate=10):
        self.a = 0  # 기울기
        self.b = 0  # y 절편
        self.data = None  # X, y 데이터프레임
        self.uprate = up_rate  # 증가량 

    def fit(self, X=None, y=None):
        if X is None or y is None:
            raise Exception("x and y cannot be None.")
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise Exception("X and y should be ndarray")
        if len(X) != len(y):
            raise Exception("X length and y length cannot be different")
        self.fit_logic(X, y)

    # 학습
    def fit_logic(self, X, y):
        up = []  # 데이터마다 증가량에 따른 기울기를 담는 리스트.

        self.data = list(zip(X, y))
        self.data = np.sort(self.data)
        self.data = pd.DataFrame(self.data, columns=["X", "y"])

        def duplicate(x):
            duplicate_data = self.data[(self.data["X"] > (x - 0.1)) & (self.data["X"] < (x + 0.1))]["y"]
            return sum(duplicate_data) / len(duplicate_data)

        self.data["y"] = self.data["X"].apply(lambda x: duplicate(x))
        self.data = self.data.drop_duplicates(["y"], keep="first")
        self.data = self.data.reset_index(drop=True)

        for i in range(self.data.shape[0]):
            try:
                up.append((self.data.iloc[i, 1] - self.data.iloc[i + self.uprate, 1]) / (
                        self.data.iloc[i, 0] - self.data.iloc[i + self.uprate, 0]))
            except:
                pass

        for i in reversed(range(self.data.shape[0])):
            try:
                up.append((self.data.iloc[i, 1] - self.data.iloc[i - self.uprate, 1]) / (
                        self.data.iloc[i, 0] - self.data.iloc[i - self.uprate, 0]))
            except:
                pass

        # 기울기 구하기
        self.a = sum(up) / len(up)

        # 절편 구하기
        self.b = self.data.iloc[0, 1] - (self.a * self.data.iloc[0, 0])

    # 예측
    def predict(self, X):
        y_pred = []
        for i in X:
            y_pred.append(self.a * i + self.b)
        return y_pred

    # 정보
    def info(self, ifpr=True):
        if ifpr:
            print(f"y = {self.a}x + ({self.b})")
        return self.a, self.b

    def evaluation_graph(self, X, y):
        plt.scatter(X, y, label="original")
        plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], color="red", label="preprocessing")

        pred = self.predict(self.data.iloc[:, 0])
        plt.plot(self.data.iloc[:, 0], pred, color="yellow", label="predict")

        plt.legend()

        plt.show()
```

