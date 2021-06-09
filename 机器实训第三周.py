import numpy as np
import random

x = np.array([[3, 3], [4, 3], [1, 1]])
y = np.array([1, 1, -1])
w = np.array([0, 0])
b = 0
n = 1
for i in range(10):
    c = random.randint(0, 2)
    y_new = y[c] * (w * x[c] + b)
    #if y_new >= 0:
      #  w = w + (n * x[c] * y[c])
       # b = b + (n * (x[c] * y[c]))
    #else:
     #   print(w)
      #  print(b)
