import numpy as np

# def step_function(x):
#     if x>0:
#         return 1
#     else:
#         return 0


def step_function(x):
    y = x > 0
    return y.astype(np.int)


x = np.array([-1.0, 1.0, 2.0])

print("x", x)
y = x > 0
print("y", y)

y = y.astype(np.int)
print("y", y)