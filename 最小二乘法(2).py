import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

Xi = np.array([2.8, 2.9, 3.2, 3.2, 3.4, 3.2, 3.3, 3.7, 3.9, 4.2, 3.9, 4.1, 4.2, 4.4, 4.2])
Yi = np.array([25, 27, 29, 32, 34, 36, 35, 39, 42, 45, 44, 44, 45, 48, 47])
plt.scatter(Xi, Yi)
plt.show()


def func(p, x):
    k, b = p
    return k * x + b


def error(p, x, y):
    return func(p, x) - y


p0 = [1, 20]
Para = leastsq(error, p0, args=(Xi, Yi))  # 使用最小二乘法快速进行拟合
k, b = Para[0]
print("k=", k, "b=", b)
print("求解的拟合直线为：")
print("y=" + str(round(k, 2)) + "x" + str(round(b, 2)))
plt.figure(figsize=(8, 6))
plt.scatter(Xi, Yi, color="blue", label="sample data", linewidth=2)
x = np.linspace(2, 5, 100) #生成2到5之间的100哥数据
y = k * x + b
plt.plot(x, y, color="red", label="Fitting straight line", linewidth=2)
plt.legend(loc="lower right")
plt.show()
