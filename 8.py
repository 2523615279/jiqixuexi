import matplotlib.pyplot as plt
import random

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def jc(x):
    mul = 1
    for i in range(len(x)):
        mul = mul * x[i]
        print("mul", mul)
    return mul


mul = jc(x)
print(mul)
