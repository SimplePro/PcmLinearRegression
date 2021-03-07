import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions_module import Functions


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
            sample_xy = sorted([xy for xy in list(zip(sample_xy.iloc[:, 0].tolist(), sample_xy.iloc[:, 1].tolist()))],
                               key=lambda x: x[0])
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


# degree 가 2 이상일 때 model 예측하는 클래스
class PimDegree2Up:

    def __init__(self, epoch=10000, dp=0.1, degree=2):
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

        # degree 에 따라 유동적으로 구역 나누기
        zone_unit = (self.data.iloc[-1, 0] - self.data.iloc[0, 0]) / (self.degree+1)  # 구역 단위        
        zones = [self.data[self.data["X"] < (self.data.iloc[0, 0] + zone_unit)]]
        
        for i in range(2, self.degree+1):
            zones.append(self.data[(self.data["X"] >= (self.data.iloc[0, 0] + zone_unit * (i-1))) & (self.data["X"] < (self.data.iloc[0, 0] + zone_unit * i))])
            
        zones.append(self.data[self.data["X"] >= (self.data.iloc[0, 0] + zone_unit * self.degree)])

        for i in range(self.epoch):
            try:
                functions = Functions()
                funcs = []

                for idx in range(len(zones)):
                    funcs.append(zones[idx].sample(n=1).iloc[0, :].tolist())

                for x, y in funcs:
                    functions.add_func((x, y))

                up.append(functions.predict_func())

            except:
                pass

        for i in up:
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


# PimDegree1 과 PimDegree2Up 를 융합한다.
class PimLinearRegression:
    def __init__(self, dp=0.1, degree=None, epoch=None):

        if dp is None:
            raise Exception("dp must not be None.")

        if degree is None:
            raise Exception("degree must not be None.")

        if epoch is None:
            raise Exception("epoch must not be None.")

        if degree == 1:
            self.model = PimDegree1(epoch=epoch, dp=dp)

        elif degree >= 2:
            if epoch < 1000:
                raise Exception("epoch must not be smaller than 1000")

            self.model = PimDegree2Up(epoch=epoch, dp=dp, degree=degree)

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
    pimDegree1 = PimLinearRegression(epoch=1000, dp=0.1, degree=1)
    X = 2 * np.random.rand(100, 1)
    y = 6 + 4 * X + np.random.randn(100, 1)

    X = np.ravel(X, order="C")
    y = np.ravel(y, order="C")

    pimDegree1.fit(X, y)
    pimDegree1.evaluation_graph(X, y)

    np.random.seed(49)

    # degree 5
    pimDegree5 = PimLinearRegression(epoch=10000, dp=0.1, degree=5)

    # data x, y
    np.random.seed(1)
    X = np.round(6 * np.random.rand(200, 1) - 3, 3)
    y = 0.5 * X ** 5 + X ** 3 + 12 + (30 * np.random.randn(200, 1))

    X = X.reshape(X.shape[0], )
    y = y.reshape(y.shape[0], )

    # fit
    pimDegree5.fit(X, y)
    print(pimDegree5.info())
    plt.title(10000)
    pimDegree5.evaluation_graph(X, y)
