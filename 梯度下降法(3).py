import numpy as np
import matplotlib.pyplot as plt


def gradient(gt_y, pred_y, x):
    N = len(x)
    diff = (1 / N) * (pred_y - gt_y)
    dw = np.dot(diff, x)
    db = np.dot(diff, np.array(np.ones(shape=(N, 1), dtype=np.float)))
    return dw, db


def train(w, b, x, gt_y, lr, max_iter):
    N = len(x)
    plt.figure()
    plt.ion()
    for num in range(max_iter):
        pred_y = np.dot(w, x) + b
        delta = pred_y - gt_y
        w = w - lr * gradient(gt_y, pred_y, x)[0]
        b = b - lr * gradient(gt_y, pred_y, x)[1]
        loss = (1 / N) * np.dot(delta, delta.T)
        plt.clf()
        plt.scatter(x, gt_y)
        plt.plot(x, pred_y, 'r-', lw=3)
        plt.xlim(2, 5)
        plt.ylim(20, 50)
        plt.title('Iteration:%d' % num)
        plt.text(4, 4, 'loss=%.4f' % loss)
        plt.text(4, 3, 'Y=%.4fx+%.4f' % (w, b))
        plt.pause(0.01)
        plt.show()
        if num % 20 == 0:
            print('Iteration:%d \tY=%.4fx+%.4f \tLoss=%.4f' % (num, w, b, loss))


if __name__ == '__main__':
    x = [2.8, 2.9, 3.2, 3.2, 3.4, 3.2, 3.3, 3.7, 3.9, 4.2, 3.9, 4.1, 4.2, 4.4, 4.2]
    gt_y = [25, 27, 29, 32, 34, 36, 35, 39, 42, 45, 44, 44, 45, 48, 47]
    train(0, 0, x, gt_y, 0.1, 2000)
