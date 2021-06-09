import numpy as np
import matplotlib.pyplot as plt

points = np.genfromtxt("货运量与工业总产值数据集.csv", delimiter=',')
x = points[:, 0]
y = points[:, 1]
plt.scatter(x, y)
plt.show()


def compute_cost(w, b, points):
    total_cost = 0
    M = len(points)
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        total_cost += (y - w * x - b) ** 2
    return total_cost / M


def fit(points):
    M = len(points)
    x_bar = np.mean(points[:, 0])
    sum_yx = 0
    sum_x2 = 0
    sum_delta = 0
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_yx += y * (x - x_bar)
        sum_x2 += x ** 2
    w = sum_yx / (sum_x2 - M * (x_bar ** 2))
    for i in range(M):
        x = points[i, 0]
        y = points[i, 1]
        sum_delta += (y - w * x)
    b = sum_delta / M
    return w, b


w, b = fit(points)
w, b
print("w is:", w)
print("b is:", b)
cost = compute_cost(w, b, points)
plt.scatter(x, y)
pred_y = w * x + b
plt.plot(x, pred_y, c='r')
plt.show()