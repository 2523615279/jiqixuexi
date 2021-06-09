# 导入库文件
import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.utils import shuffle


dataset = pd.read_csv("iris.csv")   # 读取文件，将数据放在dataset中
dataset_new = dataset.replace({'setosa': 0, 'versicolor': 1, 'virginica': 2})   # 将数据替换成numpy数据格式

# 取出数据的特征以及标签
feature = dataset_new.iloc[:, 1:5]
flag = dataset_new.iloc[:, 5]

# 将数据转换成数组类型
feature_new = np.array(feature)
flag_new = np.array(flag)

# 打乱数据
feature_new = shuffle(feature_new)
flag_new = shuffle(flag_new)

# 获取data_num的数组长度，并将长度转换成整形
data_num = len(feature_new)
train_num = int(data_num * 0.8)

# 划分特征的训练集和测试集
train_feature = feature_new[0:train_num, :]
test_feature = feature_new[train_num:, :]
# 划分标签的训练集和测试集
train_flag = flag_new[0:train_num, ]
test_flag = flag_new[train_num:, ]

# 搭建网络模型
model = Sequential()   # 通过序列模型搭建神经网络骨架
model.add(Dense(12, input_dim=4, activation='sigmoid')) # 通过add 方法添加隐藏层，这里添加的是Dense全连接层
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='relu'))

# 编译网络模型
model.compile(loss='logcosh', optimizer='SGD', metrics=['accuracy'])
# 训练网络
model.fit(x=train_feature, y=train_flag, epochs=150, batch_size=5)

# 验证网络精度
score = model.evaluate(test_feature, test_flag)
print(score)
