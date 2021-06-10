import matplotlib.pyplot as plt
import numpy as np

house_size = [91, 72, 71, 92, 102, 82, 76, 81, 101, 87]
house_price = [128, 92, 95, 117, 135, 107, 91, 118, 147, 111]

weigts = 1
bias = 10
n = 0.00001


def drawing():
    plt.xlabel("size")
    plt.ylabel("price")
    plt.axis([50, 150, 60, 180])
    plt.scatter(house_size, house_price, c='b', marker='x')
    x = np.linspace(50, 160, 100)
    y = neuron(house_size)
    plt.plot(house_size, y, c='r')
    plt.show()


def neuron(x):
    return weigts * x + bias


def calcMSE():
    house_expected = [neuron(item) for item in house_size]
    return sum([pow(house_expected[i] - house_size[i], 2) for i in range(0, len(house_size))]) / (2 * (len(house_size)))


def gengxin():
    global weigts, bias, n
    # cost_w = sum([((weigts * x[i] + bias) * x[i]) for i in range(0, len(x))]) / (len(x))
    # cost_b = sum([(weigts * x[i] + bias) for i in range(0, len(x))]) / (len(x))
    cost_w = sum((neuron(house_size[i]) - house_price[i]) * house_price[i] for i in range(0, len(house_size))) / len(house_size)
    cost_b = sum((neuron(house_size[i]) - house_price[i]) for i in range(0, len(house_size))) / len(house_size)
    w = weigts - cost_w * n
    b = bias - cost_b * n
    weigts = w
    bias = b
    print(cost_w, cost_b)
    print(w)
    return w, b


if __name__ == '__main__':
    drawing()
    for i in range(1):
        w, b = gengxin()
        cost = calcMSE()
    y = neuron(house_size)
    print(weigts, bias)
    print(y)
    drawing()
