x = []                  # 定义一个x列表
w = [0.5, 0.5]          # 定义一个有初值的w
b = -0.8                # 定义b
a = input('x=')         # 输入两个x
a = a.split(",")        # 切割符使用逗号
# 把输入的x从字符串转换成整型
for i in range(len(a)):
    x.append(int(a[i]))
# 定义y的方程并求出y
y = w[0] * x[0] + w[1] * x[1] + b
# 判断y是否大于0，是的话输出1，不是则输出0
if y > 0:
    print("1")
else:
    print("0")
