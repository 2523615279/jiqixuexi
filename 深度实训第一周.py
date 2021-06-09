import matplotlib.pyplot as plt

x = [91, 72, 71, 92, 102, 82, 76, 81, 101, 87]
y = [128, 92, 95, 117, 135, 107, 91, 118, 145, 111]


def shen(x):
    w = 1.55
    bias = -19
    return w * x + bias


def fit(x):
    for i in range(len(x)):
        m = len(x)
        y_y = shen(x[i])
        sum = sum + (y[i] - y_y) ** 2
        return sum


print(sum)
pre_y = []
for i in range(len(y)):
    pre_y.append(shen(x[i]))
plt.scatter(x, y)
plt.plot(x, pre_y, 'r')
plt.show()
