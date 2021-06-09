import numpy as np
import matplotlib.pyplot as plt

train_data = np.genfromtxt("货运量与工业总产值数据集.csv", delimiter=",", encoding='UTF-8')
x_data = train_data[:, 0]
y_data = train_data[:, 1]
plt.scatter(x_data, y_data)
plt.show()
lr = 0.1
theat0 = 0
theat1 = 0
iteators = 2000


def compute_loss(x_data, y_data, theat0, theat1):
    sum = 0
    for i in range(len(x_data)):
        sum += pow(theat0 + theat1 * x_data[i] - y_data[i], 2)
    return sum / (2 * len(x_data))


def gradient_fun(x_data, y_data, theat0, theat1, iteators, lr):
    m = float(len(x_data))
    cost = np.zeros(iteators)
    for j in range(iteators):
        sum1 = 0
        sum2 = 0
        for i in range(len(x_data)):
            sum1 += (theat0 + theat1 * x_data[i]) - y_data[i]
            sum2 += ((theat0 + theat1 * x_data[i]) - y_data[i]) * x_data[i]
        theat0 = theat0 - lr * sum1 / m
        theat1 = theat1 - lr * sum2 / m
        cost[j] = compute_loss(x_data, y_data, theat0, theat1)
        print('iteators = ', j, 'loss=', compute_loss(x_data, y_data, theat0, theat1), 'theat0', theat0, 'theat1', theat1)
    return theat0, theat1, cost


theat0, theat1, cost = gradient_fun(x_data, y_data, theat0, theat1, iteators, lr)
plt.plot(x_data, y_data, "b.")
plt.plot(x_data, theat0 + theat1 * x_data, 'r')
plt.show()
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iteators), cost, 'b')
ax.set_xlabel('Iterations')
ax.set_ylabel('cost')
ax.set_title('cost__iteators')
pre_data = np.genfromtxt("货运量与工业总产值数据集.csv", delimiter=",", encoding='UTF-8')
x_data1 = pre_data[:, 0]


def predict_func(x_data1, theat0, theat1):
    theat0 = 0.03501490948451012
    theat1 = 1.4788038980136102
    pre_y = np.zeros(len(x_data1))
    for i in range(len(x_data1)):
        pre_y[i] = theat0 + theat1 * x_data[i]
        print("x:", x_data[i], "预测值", pre_y[i])


theat0 = 0.03501490948451012
theat1 = 1.4788038980136102
predict_func(x_data1, theat0, theat1)