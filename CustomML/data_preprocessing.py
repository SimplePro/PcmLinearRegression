
# data 를 압축하는 로직을 더 예측하기 편하게 바꾼다면 정확도가 많이 향상 될 것이다.
def duplicate(x, dp, data):
    duplicate_data = data[(data["X"] > (x - dp)) & (data["X"] < (x + dp))]["y"]
    return sum(duplicate_data) / len(duplicate_data)

def compression(data, dp):

    data["y"] = data["X"].apply(lambda x: duplicate(x, dp, data))
    data = data.drop_duplicates(["y"], keep="first")
    data = data.sort_values(by=["X"], axis=0)
    data = data.reset_index(drop=True)

    return data


def zone(data, degree):
    # degree 에 따라 유동적으로 구역 나누기
    zone_unit = (data.iloc[-1, 0] - data.iloc[0, 0]) / (degree+1)  # 구역 단위
    zones = [data[data["X"] < (data.iloc[0, 0] + zone_unit)]]

    if degree >= 2:
        for i in range(2, degree+1):
            zones.append(data[(data["X"] >= (data.iloc[0, 0] + zone_unit * (i-1))) & (data["X"] < (data.iloc[0, 0] + zone_unit * i))])

    zones.append(data[data["X"] >= (data.iloc[0, 0] + zone_unit * degree)])
    return zones