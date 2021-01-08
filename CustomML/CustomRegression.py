import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PimLinearRegression():
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
