import matplotlib.pyplot as plt

x = [2.8, 2.9, 3.2, 3.2, 3.4, 3.2, 3.3, 3.7, 3.9, 4.2, 3.9, 4.1, 4.2, 4.4, 4.2]
y = [25, 27, 29, 32, 34, 36, 35, 39, 42, 45, 44, 44, 45, 48, 47]
print(type(x))
plt.scatter(x, y)
plt.show()


def avg(x):
    m = len(x)
    sum = 0
    for num in x:
        sum += num
    return sum / m


def fit(x, y):
    x_avg = avg(x)
    y_avg = avg(y)
    m = len(x)
    tmp_1 = 0
    tmp_2 = 0
    for i in range(m):
        tmp_1 += (x[i] - x_avg) * (y[i] - y_avg)
        tmp_2 += (x[i] - x_avg) ** 2
    a = tmp_1 / tmp_2
    b = y_avg - a * x_avg
    return a, b


a, b = fit(x, y)
print(a)
print(b)
pre_y = []
for i in range(len(y)):
    pre_y.append(a * x[i] + b)
plt.scatter(x, y)
plt.plot(x, pre_y, c='r')
print(pre_y)
plt.show()
