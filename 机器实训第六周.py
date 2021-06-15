import matplotlib.pyplot as plt
import numpy as np # 导入库

house_size = [91, 72, 71, 92, 102, 82, 76, 81, 101, 87] # 输入数据
house_price = [128, 92, 95, 117, 135, 107, 91, 118, 147, 111]

# 定义weigts，bias，n的值
weigts = 1
bias = 10
n = 0.00001


def drawing(init_weigts, init_bias, train_weigts, train_bias):
    """
     绘图函数，绘制更新前的线和更新后的线
    :param init_weigts:
    :param init_bias:
    :param train_weigts:
    :param train_bias:
    :return:
    """
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
    """
    前向计算函数
    :param x:
    :return: weigts * x + bias
    """
    return weigts * x + bias


def calcMSE():
    """
    损失函数
    :return: sum([pow(house_expected[i] - house_size[i], 2) for i in range(0, len(house_size))]) / (2 * (len(house_size)))
    """
    house_expected = [neuron(item) for item in house_size]
    return sum([pow(house_expected[i] - house_size[i], 2) for i in range(0, len(house_size))]) / (2 * (len(house_size)))


def gengxin():
    """
    weigts和bias的更新函数
    :return: w, b
    """
    global weigts, bias, n
    cost_w = sum((neuron(house_size[i]) - house_price[i]) * house_price[i] for i in range(0, len(house_size))) / len(
        house_size)
    cost_b = sum((neuron(house_size[i]) - house_price[i]) for i in range(0, len(house_size))) / len(house_size)
    w = weigts - cost_w * n
    b = bias - cost_b * n
    weigts = w
    bias = b
    return w, b


init_weigts = weigts  # 记录更新前的weigts和bias
init_bias = bias
eoochCost = [] # 定义两个列表
epochPrice = []
for i in range(0, 50): # 循环更新50次
    train_weigts, train_bias = gengxin()
    eoochCost.append(calcMSE()) # 记录每次的损失值
    epochPrice.append(neuron(180)) # 记录每次x=180的y的值
drawing(init_weigts, init_bias, train_weigts, train_bias)
# points_x = range(0, len(eoochCost))
# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1)
# ax.plot(points_x, eoochCost, '-x')
# ax = fig.add_subplot(1, 2, 2)
# ax.plot(points_x, epochPrice, '-g')
# plt.show()