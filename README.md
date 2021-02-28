### PIM LinearRegression (Point Inclination Mean) - 점경사평균법
-------------

설명 (Degree 1)
------------
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
self.data = self.data.reset_index(drop=True)
```

#### 2. 전처리한 데이터에서 epoch 만큼 샘플을 뽑아서 기울기를 구한다.
- 증가량으로 기울기를 구한다.
``` python
for i in range(self.epoch):
    sample_xy = self.data.sample(n=2)
    sample_xy = sorted([xy for xy in list(zip(sample_xy.iloc[:, 0].tolist(), sample_xy.iloc[:, 1].tolist()))], key=lambda x: x[0])
    up.append((sample_xy[0][1] - sample_xy[1][1]) / (sample_xy[0][0] - sample_xy[1][0]))

# 기울기 구하기
self.a = round(sum(up) / len(up), 3)
```


#### 3. 절편은 y 값과 예측값의 차이의 평균으로 구할 수 있다.
``` python
# 절편 구하기
b_lst = self.data.iloc[:, 1] - (self.a * self.data.iloc[:, 0])
self.b = round(sum(b_lst) / len(b_lst), 3)
```

#### 그렇게 y = ax + b 형태의 일차함수(LinearRegression) 그래프를 얻을 수 있다.
``` python
# 정보
def info(self, ifpr=True):
    if ifpr:
        print(f"y = {self.a}x + ({self.b})")
    return self.a, self.b
```

#### 4. X 값과 y 값으로 평가 그래프를 그릴 수 있다.
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
<img width="900" height="600" src="https://user-images.githubusercontent.com/66504341/109374688-6d8c4b00-78fa-11eb-8c4c-0996eeddc4a9.png"></img>
</div>

#### 5. 예측은 predict 메소드를 이용하여 할 수 있다.
``` python
# 예측
def predict(self, X):
    y_pred = []
    for i in X:
        y_pred.append(self.a * i + self.b)
    return y_pred
```

### PimLinearRegression Degree1 기술 정리
|스텝|설명|
|:---:|:---:|
|1|x 값이 비슷한 데이터들끼리 평균 데이터를 구하여 전반적인 전처리를 한다.|
|2|전처리한 데이터에서 샘플을 뽑아서 기울기들을 epoch 만큼 구하고, 평균을 내서 최종 기울기를 구한다.|
|3|y 값과 예측값(기울기만을 이용한)의 차이의 평균으로 절편을 구할 수 있다.|
|4|2 와 3을 이용해서 최종적인 모델을 예측해낸다.|

#### 다음과 같이 사용할 수 있다
``` python
# 임포트
from CustomML.CustomRegression import PimLinearRegression

# 생성
lr_model = PimLinearRegression(epoch=1000, dp=0.1, degree=1)

# 학습
lr_model.fit(heights, weights)

# 실제 값의 분포와 선형회귀를 그래프로 표현.
lr_model.evaluation_graph(heights, weights)
```

설명 (degree 2)
---------
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

#### 2. 샘플 데이터를 뽑아서 함수를 예측할 때 고르게 샘플 데이터를 뽑기 위하여 데이터를 세개의 구역으로 나눈다.
``` python
zone_unit = self.data.shape[0] // 3   # 구역 단위
# 데이터의 구역을 나누는 이유는 sample 3 개를 뽑았을 때에 비슷한 구역에서 3 개를 뽑아서 함수를 예측하게 되면 불필요한 함수가 예측될 수 있기 때문이다.

data1 = self.data.iloc[0:zone_unit, :]  # 구역 1
data2 = self.data.iloc[zone_unit:zone_unit*2, :]  # 구역 2
data3 = self.data.iloc[zone_unit*2:, :]  # 구역 3
```

#### 3. 세 개의 샘플 데이터를 뽑아서 함수들을 예측한다. (epoch 만큼 반복)
``` python
for i in range(self.epoch):
    try:
        functions = Functions(scale=self.scale)
        func1 = data1.sample(n=1)
        func2 = data2.sample(n=1)
        func3 = data3.sample(n=1)
        funcs = [func1.iloc[0, :].tolist(), func2.iloc[0, :].tolist(), func3.iloc[0, :].tolist()]

        for x, y in funcs:
            functions.add_func((x, y))

        up.append(functions.predict_func()[0])

    except:
        pass

a = []
b = []
c = []

for i in up:
    a.append(i["a"])
    b.append(i["b"])
    c.append(i["c"])

self.a = sum(a) / len(a)
self.b = sum(b) / len(b)
self.c = sum(c) / len(c)
```

#### 그렇게 y = ax^2 + bx + c 형태의 이차함수(LinearRegression) 그래프를 얻을 수 있다.
``` python
# 정보
def info(self, ifpr=True):
    if ifpr:
    print(f"y = {self.a}x^2 + ({self.b}x) + ({self.c})")

    return self.a, self.b, self.c
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
<img width="900" height="600" src="https://user-images.githubusercontent.com/66504341/109407579-48b0da00-79c5-11eb-8583-66ac8a77c6c4.png"></img>
</div>

#### 5. 예측은 predict 메소드를 이용하여 할 수 있다.
``` python
# 예측
def predict(self, X):
    y_pred = []

    for i in X:
        y_pred.append((self.a * (i ** 2)) + (self.b * i) + self.c)

    return y_pred
```

### PimLinearRegression Degree2 기술 정리
|스텝|설명|
|:---:|:---:|
|1|x 값이 비슷한 데이터들끼리 평균 데이터를 구하여 전반적인 전처리를 한다.|
|2|샘플 데이터를 고르게 뽑기 위하여 데이터를 세개의 구역으로 나눈다.|
|3|세 개의 샘플 데이터를 뽑아서 이차함수를 예측한다. (epoch 만큼 반복)|
|4|3 에서 예측했던 함수들의 평균을 내서 최종 모델을 예측해낸다.|

#### 다음과 같이 사용할 수 있다
``` python
# 임포트
from CustomML.CustomRegression import PimLinearRegression

pimDegree2 = PimLinearRegression(epoch=10000, dp=0.1, degree=2, scale=3)

# fit
pimDegree2.fit(X, y)
print(pimDegree2.info())
pimDegree2.evaluation_graph(X, y)
```


전체코드
-----------

``` python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from derivative_pattern import Functions


# degree 가 1일 때 model 예측하는 클래스
class PimDegree1:
    def __init__(self, epoch=1000, dp=0.1):
        if epoch is None:
            raise Exception("epoch must not be None.")

        if dp is None:
            raise Exception("dp must not be None.")

        self.a = 0  # 기울기
        self.b = 0  # y 절편
        self.data = None  # X, y 데이터프레임
        self.epoch = epoch  # 반복 횟수
        self.dp = dp  # 데이터 전처리 단위

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
        self.data = pd.DataFrame(self.data, columns=["X", "y"])

        def duplicate(x):
            duplicate_data = self.data[(self.data["X"] > (x - self.dp)) & (self.data["X"] < (x + self.dp))]["y"]
            return sum(duplicate_data) / len(duplicate_data)

        self.data["y"] = self.data["X"].apply(lambda x: duplicate(x))
        self.data = self.data.drop_duplicates(["y"], keep="first")
        self.data = self.data.reset_index(drop=True)

        for i in range(self.epoch):
            sample_xy = self.data.sample(n=2)
            sample_xy = sorted([xy for xy in list(zip(sample_xy.iloc[:, 0].tolist(), sample_xy.iloc[:, 1].tolist()))], key=lambda x: x[0])
            up.append((sample_xy[0][1] - sample_xy[1][1]) / (sample_xy[0][0] - sample_xy[1][0]))

        # 기울기 구하기
        self.a = round(sum(up) / len(up), 3)

        # 절편 구하기
        b_lst = self.data.iloc[:, 1] - (self.a * self.data.iloc[:, 0])
        self.b = round(sum(b_lst) / len(b_lst), 3)

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
        plt.plot(self.data.iloc[:, 0], pred, color="yellow", label="predict", linewidth=3.0)

        plt.legend()

        plt.show()


# degree 가 2일 때 model 예측하는 클래스
class PimDegree2:

    def __init__(self, epoch=10000, dp=0.1, scale=None):
        if epoch is None:
            raise Exception("epoch must not be None.")

        if dp is None:
            raise Exception("dp must not be None.")

        if scale is None:
            raise Exception("scale must not be None.")

        self.a = 0  # ax^2 + bx + c
        self.b = 0  # ax^2 + bx + c
        self.c = 0  # ax^2 + bx + c
        self.data = None  # X, y 데이터프레임
        self.epoch = epoch  # 반복횟수
        self.dp = dp  # 데이터 전처리 단위
        self.scale = scale + 1  # 데이터의 최대 소수점 자리수
        # 이차함수 그래프를 그릴 떄에는 scale 을 입력받았을 떄. 그 scale 에 +1 을 하여. 단위를 상승시킨다.
        # (랜덤으로 점을 잡아서 이차함수를 예측할 때 scale 을 기준으로 2 이상 차이가 나야 하기 때문이다. scale 을 +1 로 잡으면 차이는 최소가 10으로 줄어들게 된다.)

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
        up = []  # 데이터마다 증가량에 따른 기울기를 담는 리스트.

        self.data = list(zip(X, y))
        self.data = pd.DataFrame(self.data, columns=["X", "y"])

        def duplicate(x):
            duplicate_data = self.data[(self.data["X"] > (x - self.dp)) & (self.data["X"] < (x + self.dp))]["y"]
            return sum(duplicate_data) / len(duplicate_data)

        self.data["y"] = self.data["X"].apply(lambda x: duplicate(x))
        self.data = self.data.drop_duplicates(["y"], keep="first")
        self.data = self.data.sort_values(by=["X"], axis=0)
        self.data = self.data.reset_index(drop=True)

        zone_unit = self.data.shape[0] // 3   # 구역 단위
        # 데이터의 구역을 나누는 이유는 sample 3 개를 뽑았을 때에 비슷한 구역에서 3 개를 뽑아서 함수를 예측하게 되면 불필요한 함수가 예측될 수 있기 때문이다.

        data1 = self.data.iloc[0:zone_unit, :]  # 구역 1
        data2 = self.data.iloc[zone_unit:zone_unit*2, :]  # 구역 2
        data3 = self.data.iloc[zone_unit*2:, :]  # 구역 3

        for i in range(self.epoch):
            try:
                functions = Functions(scale=self.scale)
                func1 = data1.sample(n=1)
                func2 = data2.sample(n=1)
                func3 = data3.sample(n=1)
                funcs = [func1.iloc[0, :].tolist(), func2.iloc[0, :].tolist(), func3.iloc[0, :].tolist()]

                for x, y in funcs:
                    functions.add_func((x, y))

                up.append(functions.predict_func()[0])

            except:
                pass

        a = []
        b = []
        c = []

        for i in up:
            a.append(i["a"])
            b.append(i["b"])
            c.append(i["c"])

        self.a = sum(a) / len(a)
        self.b = sum(b) / len(b)
        self.c = sum(c) / len(c)

    # 예측
    def predict(self, X):
        y_pred = []

        for i in X:
            y_pred.append((self.a * (i ** 2)) + (self.b * i) + self.c)

        return y_pred

    # 정보
    def info(self, ifpr=True):
        if ifpr:
            print(f"y = {self.a}x^2 + ({self.b}x) + ({self.c})")

        return self.a, self.b, self.c

    # 평가
    def evaluation_graph(self, X, y):
        plt.scatter(X, y, label="original")
        plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], color="red", label="preprocessing")

        pred = self.predict(self.data.iloc[:, 0])
        argsort = np.argsort(self.data.iloc[:, 0].tolist())
        plt.plot(self.data.iloc[:, 0][argsort].tolist(), np.array(pred)[argsort], color="yellow", label="predict", linewidth=3.0)

        plt.legend()

        plt.show()


# PimDegree1 과 PimDegree2 를 상속받아 융합한다.
class PimLinearRegression:
    def __init__(self, dp=0.1, scale=None, degree=None, epoch=None):

        if dp is None:
            raise Exception("dp must not be None.")

        if degree is None:
            raise Exception("degree must not be None.")

        if degree == 2 and scale is None:
            raise Exception("scale must not be None.")

        if epoch is None:
            raise Exception("epoch must not be None.")

        if degree == 1:
            self.model = PimDegree1(epoch=epoch, dp=dp)

        if degree == 2:
            if epoch < 1000:
                raise Exception("epoch must not be smaller than 1000")

            self.model = PimDegree2(epoch=epoch, dp=dp, scale=scale)

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
    def info(self, ifpr=True):
        return self.model.info(ifpr=ifpr)

    def evaluation_graph(self, X, y):
        self.model.evaluation_graph(X, y)


if __name__ == '__main__':

    # degree = 1
    pimDegree1 = PimLinearRegression(epoch=1000, dp=0.1, degree=1)
    X = 2 * np.random.rand(100, 1)
    y = 6 + 4 * X+np.random.randn(100, 1)

    X = np.ravel(X, order="C")
    y = np.ravel(y, order="C")

    pimDegree1.fit(X, y)
    pimDegree1.evaluation_graph(X, y)

    # degree 2
    epochs = [10000]
    # epochs.extend([1000, 5000, 10000, 20000, 30000, 40000, 50000])

    for i in epochs:
        pimDegree2 = PimLinearRegression(epoch=i, dp=0.1, degree=2, scale=3)

        # data x, y
        X = np.round(np.random.randn(100, 1), 3)
        y = (-3 * X ** 2) + (3*X) + 3 + (2.5*np.random.randn(100, 1))

        X = np.ravel(X, order="C")
        y = np.ravel(y, order="C")

        # fit
        pimDegree2.fit(X, y)
        print(pimDegree2.info())
        plt.title(i)
        pimDegree2.evaluation_graph(X, y)

```

