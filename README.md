### PCM LinearRegression (Point Coefficient Mean) - 점계수평균법
-------------
설명
---------
### PcmLinearRegression 기술 정리
|스텝|설명|
|:---:|:---:|
|1|x 값이 비슷한 데이터들끼리 평균 데이터를 구하여 전반적인 전처리를 한다.|
|2|샘플 데이터를 고르게 뽑기 위하여 데이터를 n+1개의 구역으로 나눈다.|
|3|n+1개의 샘플 데이터를 뽑아서 n차함수를 예측한다. (epoch 만큼 반복)|
|4|3 에서 예측했던 함수들의 평균을 내서 최종 모델을 예측해낸다.|


#### 1. 먼저 학습을 시작하기 전에 데이터들을 전처리한다. (preprocessing)
- x 값이 비슷한 y 값들끼리의 평균값을 구한다.  
``` python
self.data = list(zip(X, y))
self.data = pd.DataFrame(self.data, columns=["X", "y"])

def duplicate(x):
    duplicate_data = self.data[(self.data["X"] > (x - self.dp)) & (self.data["X"] < (x + self.dp))]["y"]
    return sum(duplicate_data) / len(duplicate_data)

self.data["y"] = self.data["X"].apply(lambda x: duplicate(x))
self.data = self.data.drop_duplicates(["y"], keep="first")
self.data = self.data.sort_values(by=["X"], axis=0)
self.data = self.data.reset_index(drop=True)
```

#### 2. 샘플 데이터를 뽑아서 함수를 예측할 때 고르게 샘플 데이터를 뽑기 위하여 데이터를 n+1개의 구역으로 나눈다.
``` python
# degree 에 따라 유동적으로 구역 나누기
zone_unit = (self.data.iloc[-1, 0] - self.data.iloc[0, 0]) / (self.degree+1)  # 구역 단위
zones = [self.data[self.data["X"] < (self.data.iloc[0, 0] + zone_unit)]]

if self.degree >= 2:
    for i in range(2, self.degree+1):
        zones.append(self.data[(self.data["X"] >= (self.data.iloc[0, 0] + zone_unit * (i-1))) & (self.data["X"] < (self.data.iloc[0, 0] + zone_unit * i))])

zones.append(self.data[self.data["X"] >= (self.data.iloc[0, 0] + zone_unit * self.degree)])
```

#### 3. n+1 개의 샘플 데이터를 뽑아서 함수들을 예측한다. (epoch 만큼 반복)
``` python
for i in range(self.epoch):
    functions = Functions()
    funcs = []

    for idx in range(len(zones)):
        funcs.append(zones[idx].sample(n=1).iloc[0, :].tolist())

    for x, y in funcs:
        functions.add_func((x, y))

    up.append(functions.predict_func())

for i in up:
    for j in range(len(i)):
        self.coefficients[j].append(i[j])

for i in range(len(self.coefficients)):
    self.coefficients[i] = sum(self.coefficients[i]) / len(self.coefficients[i])
```

#### n차함수(LinearRegression) 그래프를 얻을 수 있다.
``` python
# 정보
def info(self):
    return self.coefficients
```

#### 4. X 값과 y 값으로 평가 그래프를 그릴 수 있다.
``` python
def evaluation_graph(self, X, y):
    plt.scatter(X, y, label="original")
    plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], color="red", label="preprocessing")

    pred = self.predict(self.data.iloc[:, 0])
    argsort = np.argsort(self.data.iloc[:, 0].tolist())
    plt.plot(self.data.iloc[:, 0][argsort].tolist(), np.array(pred)[argsort], color="yellow", label="predict", linewidth=3.0)

    plt.legend()

    plt.show()
```

<div>
<img width="900" height="600" src="https://user-images.githubusercontent.com/66504341/110206012-b32eb200-7ebe-11eb-8679-c57daab869d3.png"></img>
</div>

#### 5. 예측은 predict 메소드를 이용하여 할 수 있다.
``` python
# 예측
def predict(self, X):
    y_pred = []

    for i in X:
        result = 0
        for idx, coe in enumerate(self.coefficients):
            result += coe * (i ** (len(self.coefficients) - idx - 1))

        y_pred.append(result)

    return y_pred
```

#### 다음과 같이 사용할 수 있다
``` python
# 임포트
from CustomML.CustomRegression import PcmLinearRegression

pcmDegree2 = PcmLinearRegression(epoch=10000, dp=0.1, degree=2)

# fit
pcmDegree2.fit(X, y)
print(pcmDegree2.info())
pcmDegree2.evaluation_graph(X, y)
```


전체코드
-----------

``` python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions_module import Functions


# PcmLinearRegression Logic class
class PcmLinearRegressionLogic:

    def __init__(self, epoch=10000, dp=0.1, degree=1):
        if epoch is None:
            raise Exception("epoch must not be None.")

        if dp is None:
            raise Exception("dp must not be None.")

        self.data = None  # X, y 데이터프레임
        self.epoch = epoch  # 반복횟수
        self.dp = dp  # 데이터 전처리 단위

        self.degree = degree  # 차수
        self.coefficients = [[] for _ in range(self.degree+1)]  # 함수의 계수들. 차수가 높은 순으로 나열함.

    # 학습
    def fit(self, X, y):
        if X is None or y is None:
            raise Exception("x and y cannot be None.")

        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise Exception("X and y should be ndarray")

        if len(X) != len(y):
            raise Exception("X length and y length cannot be different")

        self.fit_logic(X, y)

    # 학습 로직
    def fit_logic(self, X, y):
        all_coefficients = []  # 예측된 함수들의 계수를 담는 리스트.

        self.data = list(zip(X, y))
        self.data = pd.DataFrame(self.data, columns=["X", "y"])

        def duplicate(x):
            duplicate_data = self.data[(self.data["X"] > (x - self.dp)) & (self.data["X"] < (x + self.dp))]["y"]
            return sum(duplicate_data) / len(duplicate_data)

        self.data["y"] = self.data["X"].apply(lambda x: duplicate(x))
        self.data = self.data.drop_duplicates(["y"], keep="first")
        self.data = self.data.sort_values(by=["X"], axis=0)
        self.data = self.data.reset_index(drop=True)

        # degree 에 따라 유동적으로 구역 나누기
        zone_unit = (self.data.iloc[-1, 0] - self.data.iloc[0, 0]) / (self.degree+1)  # 구역 단위
        zones = [self.data[self.data["X"] < (self.data.iloc[0, 0] + zone_unit)]]

        if self.degree >= 2:
            for i in range(2, self.degree+1):
                zones.append(self.data[(self.data["X"] >= (self.data.iloc[0, 0] + zone_unit * (i-1))) & (self.data["X"] < (self.data.iloc[0, 0] + zone_unit * i))])

        zones.append(self.data[self.data["X"] >= (self.data.iloc[0, 0] + zone_unit * self.degree)])

        for i in range(self.epoch):
            functions = Functions()
            funcs = []

            for idx in range(len(zones)):
                funcs.append(zones[idx].sample(n=1).iloc[0, :].tolist())

            for x, y in funcs:
                functions.add_func((x, y))

            all_coefficients.append(functions.predict_func())

        for i in all_coefficients:
            for j in range(len(i)):
                self.coefficients[j].append(i[j])

        for i in range(len(self.coefficients)):
            self.coefficients[i] = sum(self.coefficients[i]) / len(self.coefficients[i])

    # 예측
    def predict(self, X):
        y_pred = []

        for i in X:
            result = 0
            for idx, coe in enumerate(self.coefficients):
                result += coe * (i ** (len(self.coefficients) - idx - 1))

            y_pred.append(result)

        return y_pred

    # 정보
    def info(self):
        return self.coefficients

    # 평가
    def evaluation_graph(self, X, y):
        plt.scatter(X, y, label="original")
        plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], color="red", label="preprocessing")

        pred = self.predict(self.data.iloc[:, 0])
        argsort = np.argsort(self.data.iloc[:, 0].tolist())
        plt.plot(self.data.iloc[:, 0][argsort].tolist(), np.array(pred)[argsort], color="yellow", label="predict",
                 linewidth=3.0)

        plt.legend()

        plt.show()


# PcmLinearRegression
class PcmLinearRegression:
    def __init__(self, dp=0.1, degree=None, epoch=None):

        if dp is None:
            raise Exception("dp must not be None.")

        if degree is None:
            raise Exception("degree must not be None.")

        if epoch is None:
            raise Exception("epoch must not be None.")

        self.model = PcmLinearRegressionLogic(epoch=epoch, dp=dp, degree=degree)

    # 학습
    def fit(self, X=None, y=None):
        if X is None or y is None:
            raise Exception("x and y cannot be None.")

        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise Exception("X and y should be ndarray")

        if len(X) != len(y):
            raise Exception("X length and y length cannot be different")

        self.model.fit(X, y)

    # 예측
    def predict(self, X):
        return self.model.predict(X)

    # 정보
    def info(self):
        return self.model.info()

    def evaluation_graph(self, X, y):
        self.model.evaluation_graph(X, y)


if __name__ == '__main__':

    # degree = 1
    PcmDegree1 = PcmLinearRegression(epoch=1000, dp=0.1, degree=1)
    X = 2 * np.random.rand(100, 1)
    y = 6 + 4 * X + np.random.randn(100, 1)

    X = np.ravel(X, order="C")
    y = np.ravel(y, order="C")

    PcmDegree1.fit(X, y)
    PcmDegree1.evaluation_graph(X, y)

    np.random.seed(49)

    # degree 5
    PcmDegree5 = PcmLinearRegression(epoch=10000, dp=0.1, degree=5)

    # data x, y
    np.random.seed(1)
    X = np.round(6 * np.random.rand(200, 1) - 3, 3)
    y = 0.5 * X ** 5 + X ** 3 + 12 + (30 * np.random.randn(200, 1))

    X = X.reshape(X.shape[0], )
    y = y.reshape(y.shape[0], )

    # fit
    PcmDegree5.fit(X, y)
    print(PcmDegree5.info())
    plt.title(10000)
    PcmDegree5.evaluation_graph(X, y)


```

