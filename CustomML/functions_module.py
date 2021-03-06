import numpy as np


class Functions:

    def __init__(self, scale=0):
        self.func = []  # 함숫값을 담는 변수
        self.alphabet = [chr(c).lower() for c in range(ord('A'), ord('Z')+1)]

        if scale < 0:
            raise Exception("scale must not be smaller than 0")
        self.scale = scale  # 소수점 자리수를 의미함. 예) scale = 0 이면 정수만 다룰 수 있음. scale = 1 이면 소수점 1의 자리수까지 다룰 수 있음

        self.expressions = {}

    # function 을 추가하는 메소드.
    def add_func(self, f):
        self.func.append(f)

    # 원래의 이차함수 식을 예측하는 메소드.
    # f(x) = ax^2 + bx + c
    def predict_func(self):

        """
        degree 에 따라 유동적으로 함수를 예측할 수 있도록, n 차 연립방정식에 대한 메소드를 개발해야 한다.
        """

        eqs = []
        equals = []
        for func in self.func:
            eq = []
            for i in range(len(self.func)):
                if i + 1 == len(self.func):
                    eq.append(1)
                else:
                    eq.append(func[0]**(len(self.func) - i - 1))
            eqs.append(eq)
            equals.append(func[1])

        eqs = np.array(eqs)
        equals = np.array(equals)

        result = np.linalg.solve(eqs, equals)
        # for i in range(len(result)):
        #     result[i] = round(result[i], 14)

        return result


if __name__ == '__main__':
    f1 = (1, 7)
    f2 = (-2, -32)
    f3 = (5, 395)
    f4 = (4, 208)

    functions = Functions(scale=3)  # scale 정의
    functions.add_func(f1)  # 첫번째 함숫값 추가
    functions.add_func(f2)  # 두번째 함숫값 추가
    functions.add_func(f3)  # 세번째 함숫값 추가
    functions.add_func(f4)  # 네번째 함숫값 추가

    print(functions.predict_func())  # 함수식의 계수들이 반환됨.
