import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions_module import Functions
from data_preprocessing import compression, zone


# PcmLinearRegression Logic class
class PcmLinearRegressionLogicOneVariable:

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

        all_coefficients = []  # 예측된 함수들의 계수를 담는 리스트.

        self.data = list(zip(X, y))
        self.data = pd.DataFrame(self.data, columns=["X", "y"])

        # 데이터 압축
        self.data = compression(data = self.data, dp = self.dp)
        
        # 구역 나누기
        zones = zone(data=self.data, degree=self.degree)

        # 함수 식 예측
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

        self.model = PcmLinearRegressionLogicOneVariable(epoch=epoch, dp=dp, degree=degree)

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
    PcmDegree5 = PcmLinearRegression(epoch=10000, dp=0.5, degree=5)

    # data x, y
    np.random.seed(1)
    X = np.round(6 * np.random.rand(200, 1) - 3, 3)
    y = X ** 5 - X + 12 + (40 * np.random.randn(200, 1))

    X = X.reshape(X.shape[0], )
    y = y.reshape(y.shape[0], )

    # fit
    PcmDegree5.fit(X, y)
    print(PcmDegree5.info())
    PcmDegree5.evaluation_graph(X, y)
