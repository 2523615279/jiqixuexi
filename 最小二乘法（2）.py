#
# 引入依赖
#
import numpy as np                                          # 导入numpy库函数(科学计算库)
import matplotlib.pyplot as plt                             # 导入matplotlib库函数（绘图库）
from scipy.optimize import leastsq                          # 导入scipy库函数

#
# 导入数据
#
Xi = np.array([2.8, 2.9, 3.2, 3.2, 3.4, 3.2, 3.3, 3.7, 3.9, 4.2, 3.9, 4.1, 4.2, 4.4, 4.2])  # 样本数据(Xi,Yi)，需要转换成数组(列表)形式
Yi = np.array([25, 27, 29, 32, 34, 36, 35, 39, 42, 45, 44, 44, 45, 48, 47])
'''
    设定拟合函数和偏差函数
    函数的形状确定过程：
    1.先画样本图像
    2.根据样本图像大致形状确定函数形式(直线、抛物线、正弦余弦等)
'''
#
# 画出图像
#
plt.scatter(Xi, Yi)
plt.show()

#
# 设置拟合函数
#
def func(p, x):                                             # 指定函数的形状
    k, b = p
    return k*x+b

#
# 设置偏差函数
#
def error(p, x, y):                                         # x,y都是列表:这里的x,y更上面的Xi,Yi中是一一对应的
    return func(p, x)-y
'''
    主要部分：附带部分说明
    1.leastsq函数的返回值tuple，第一个元素是求解结果，第二个是求解的代价值
    2.（第二个值）：Value of the cost function at the solution
    3.实例：Para=>(array([ 0.61349535,  1.79409255]), 3)
    4.返回值元组中第一个值的数量跟需要求解的参数的数量一致
'''
p0 = [1, 20]                                                # k,b的初始值，可以任意设定,经过几次试验，发现p0的值会影响cost的值：Para[1]
Para = leastsq(error, p0, args=(Xi, Yi))                    # 把error函数中除了p0以外的参数打包到args中(使用要求)
k, b = Para[0]

print("k=", k, "b=", b)                                     # 读取结果
print("求解的拟合直线为:")
print("y="+str(round(k, 2))+"x+"+str(round(b, 2)))
'''
   绘图，看拟合效果.
   matplotlib默认不支持中文，label设置中文的话需要另行设置
   如果报错，改成英文就可以
'''
plt.figure(figsize=(8, 6))                                  # 指定图像比例： 8：6
plt.scatter(Xi, Yi, color="blue", label="sample data", linewidth=2)
x = np.linspace(2, 5, 100)
y = k*x+b
plt.plot(x, y, color="red", label="Fitting straight line", linewidth=2)
plt.legend(loc='lower right')                               # 绘制图例
plt.show()