import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from derivative_pattern import Functions


# degree 가 1일 때 model 예측하는 클래스
class PimDegree1:
    def __init__(self, up_rate=10, dp=0.1):
        if up_rate is None:
            raise Exception("up_rate must not be None.")
        if dp is None:
            raise Exception("dp must not be None.")

        self.a = 0  # 기울기
        self.b = 0  # y 절편
        self.data = None  # X, y 데이터프레임
        self.uprate = up_rate  # 증가량
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

    def __init__(self, up_rate=10, dp=0.1, scale=None):
        if up_rate is None:
            raise Exception("up_rate must not be None.")

        if dp is None:
            raise Exception("dp must not be None.")

        if scale is None:
            raise Exception("scale must not be None.")

        self.a = 0  # ax^2 + bx + c
        self.b = 0  # ax^2 + bx + c
        self.c = 0  # ax^2 + bx + c
        self.data = None  # X, y 데이터프레임
        self.uprate = up_rate  # 증가량
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
        self.data = self.data.reset_index(drop=True)

        for i in range(self.data.shape[0]):
            try:
                functions = Functions(scale=self.scale)
                functions.add_func((self.data.iloc[i, 0], self.data.iloc[i, 1]))
                functions.add_func((self.data.iloc[i + self.uprate, 0], self.data.iloc[i + self.uprate, 1]))
                functions.add_func((self.data.iloc[i + self.uprate*2, 0], self.data.iloc[i + self.uprate*2, 1]))
                functions.h()
                functions.t()
                up.append(functions.predict_func()[0])

            except:
                pass

        for i in reversed(range(self.data.shape[0])):
            try:
                functions = Functions(scale=self.scale)
                functions.add_func((self.data.iloc[i, 0], self.data.iloc[i, 1]))
                functions.add_func((self.data.iloc[i - self.uprate, 0], self.data.iloc[i - self.uprate, 1]))
                functions.add_func((self.data.iloc[i - self.uprate * 2, 0], self.data.iloc[i - self.uprate * 2, 1]))
                functions.h()
                functions.t()
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
            y_pred.append((self.a * i ** 2) + (self.b * i) + self.c)

        return y_pred

    # 정보
    def info(self, ifpr=True):
        if ifpr:
            print(f"y = {self.a}x^2 + ({self.b}x) + ({self.c})")
        return self.a, self.b, self.c

    def evaluation_graph(self, X, y):
        plt.scatter(X, y, label="original")
        plt.scatter(self.data.iloc[:, 0], self.data.iloc[:, 1], color="red", label="preprocessing")

        pred = self.predict(self.data.iloc[:, 0])
        pred_argsort = np.argsort(pred)
        plt.plot(self.data.iloc[:, 0][pred_argsort], np.array(pred)[pred_argsort], color="yellow", label="predict", linewidth=3.0)

        plt.legend()

        plt.show()


# PimDegree1 과 PimDegree2 를 상속받아 융합한다.
class PimLinearRegression:
    def __init__(self, up_rate=10, dp=0.1, scale=None, degree=None):
        if up_rate is None:
            raise Exception("up_rate must not be None.")

        if dp is None:
            raise Exception("dp must not be None.")

        if degree is None:
            raise Exception("degree must not be None.")

        if degree == 2 and scale is None:
            raise Exception("scale must not be None.")

        if degree == 1:
            self.model = PimDegree1(up_rate=up_rate, dp=dp)

        if degree == 2:
            self.model = PimDegree2(up_rate=up_rate, dp=dp, scale=scale)

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
