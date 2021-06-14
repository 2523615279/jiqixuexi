import matplotlib.pyplot as plt
import numpy as np

house_size = [91, 72, 71, 92, 102, 82, 76, 81, 101, 87]
house_price = [128, 92, 95, 117, 135, 107, 91, 118, 147, 111]

weigts = 1
bias = 10
n = 0.00001


def drawing(init_weigts, init_bias, train_weigts, train_bias):
    plt.xlabel("size")
    plt.ylabel("price")
    plt.axis([50, 150, 60, 180])
    x = np.linspace(50, 160, 100)
    plt.scatter(x=house_size, y=house_price, c='b', marker='x')
    pointe_x = [50, 160]
    plt.plot(pointe_x, list(item * init_weigts + init_bias for item in pointe_x), '-x')
    plt.plot(pointe_x, list(item * train_weigts + train_bias for item in pointe_x), '-g')
    plt.show()


def neuron(x):
    return weigts * x + bias


def calcMSE():
    house_expected = [neuron(item) for item in house_size]
    return sum([pow(house_expected[i] - house_size[i], 2) for i in range(0, len(house_size))]) / (2 * (len(house_size)))


def gengxin():
    global weigts, bias, n
    cost_w = sum((neuron(house_size[i]) - house_price[i]) * house_price[i] for i in range(0, len(house_size))) / len(house_size)
    cost_b = sum((neuron(house_size[i]) - house_price[i]) for i in range(0, len(house_size))) / len(house_size)
    w = weigts - cost_w * n
    b = bias - cost_b * n
    weigts = w
    bias = b
    return w, b

init_weigts = weigts
init_bias = bias
for i in range(0, 100):
    train_weigts, train_bias = gengxin()
drawing(init_weigts, init_bias, train_weigts, train_bias)