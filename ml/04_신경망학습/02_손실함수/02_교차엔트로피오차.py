import numpy as np

def cross_entropy_error(y,t):
    print("t :", t, "y : ", y)
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

t = [0,0,1,0,0,0,0,0,0,0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print("---")
print(cross_entropy_error(np.array(y), np.array(t)))


print("----")
print(cross_entropy_error(np.array([0,0.9,0.1,0]), np.array([0, 1, 0, 0])))
print(cross_entropy_error(np.array([0,0.8,0.2,0]), np.array([0, 1, 0, 0])))

print("--- 여기까지 그냥 교차 엔트로피 --- ")

def cross_entropy_error4batch(y, t):
    if y.ndim == 1:
        print("차원이 1인 경우 리쉐이프", t)
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        print("리쉐이프된 결과 :", t);

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        print("사이즈가 같다고.?")
        t = t.argmax(axis=1)
        print("after t :", t)

    batch_size = y.shape[0]
    print("배치 사이즈 :", batch_size)
    print("np arrange" , np.arange(batch_size))
    z = y[np.arange(batch_size), t]
    print("z:", z)
    return -np.sum(np.log(z)) / batch_size

print(cross_entropy_error4batch(np.array([0,0.9,0.1,0]), np.array([0, 1, 0, 0])))

print("--- 두번째 여러개 배치입니다.. ----")
print(cross_entropy_error4batch(
    np.array([ [0,0.9,0.1,0], [0, 0.8, 0.5, 0] ]),
    np.array([ [0, 1, 0, 0] , [0, 0, 1, 0]])))


print("--- 두번째 여러개 배치입니다.. ----")
