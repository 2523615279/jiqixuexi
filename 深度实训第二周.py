import numpy as np
from keras.models import Sequential
from keras.layers import Dense

data = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

# 将数据划分为特征以及标签
train_feature = data[0:650, 0:8]
train_flag = data[0:650, 8]
test_feature = data[650: 768, 0:8]
test_flag = data[650:768, 8]
print(test_flag)

model = Sequential()
model.add(Dense(32, input_dim=8, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(26, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(22, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
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

# 训练过程

model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
model.fit(x=train_feature, y=train_flag, epochs=150, batch_size=10)

score = model.evaluate(test_feature, test_flag)
print(score)