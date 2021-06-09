import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt("货运量与工业总产值数据集.csv", delimiter=",", encoding='UTF-8')
x_data = data[:, 0]
y_data = data[:, 1]
plt.scatter(x_data, y_data)
plt.show()
lr = 0.1
b = 0
k = 0
epochs = 2000


def compute_error(b, k, x_data, y_data):
    totalError = 0
    for i in range(0, len(x_data)):
        totalError += (y_data[i] - (k * x_data[i] + b)) ** 2
    return totalError / float(len(x_data)) / 2.0

def gradient_descent_runner(x_data, y_data, b, k, lr, epochs):
    m = float(len(x_data))
    for i in range(epochs):
        b_grad = 0
        k_grad = 0
        for j in range(0, len(x_data)):
            b_grad += (1 / m) * (((k * x_data[j]) + b) - y_data[j])
            k_grad += (1 / m) * x_data[j] * (((k * x_data[j]) + b) - y_data[j])
        b = b - (lr * b_grad)
        k = k - (lr * k_grad)
    return b, k


print("Starting b = {0},k ={1},error = {2}".format(b, k, compute_error(b, k, x_data, y_data)))
print("running..")
b, k = gradient_descent_runner(x_data, y_data, b, k, lr, epochs)
print("after {0} iterations b = {1},k={2},error={3}".format(epochs, b, k, compute_error(b, k, x_data, y_data)))
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, k * x_data + b, 'r')
print(k)
print(x_data)
print(b)

plt.show()
