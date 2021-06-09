from sklearn import datasets
from keras.utils import to_categorical  # 导入独热编码方法
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.models import model_from_json

dataset = datasets.load_iris()
feature = dataset.data  # 将特征取出
flag = dataset.target  # 将标签取出

train_feature, test_feature, train_flag, test_flag = train_test_split(feature, flag, train_size=0.2)
# 将普通话类别数据转换为独热编码形式
train_flag_labels = to_categorical(train_flag, num_classes=3)
test_flag_labels = to_categorical(test_flag, num_classes=3)


def create_model(optimizer='rmsprop', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(12, input_dim=4, activation='relu', kernel_initializer=init))  # kernel_initializer权重初始化的方法
    model.add(Dense(16, activation='relu', kernel_initializer=init))
    model.add(Dense(3, activation='softmax', kernel_initializer=init))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


model = create_model()
model.fit(train_feature, train_flag_labels, epochs=200, batch_size=10)
score = model.evaluate(test_feature, test_flag_labels)
# 以json格式保存网络模型
# 保存网络结构
model_json = model.to_json()
with open('model.json', 'w') as file:
    file.write(model_json)
# 保存网络参数
model.save_weights('model.json.h5')

model_yaml = model.to_yaml()
with open('model.yaml', 'w') as file:
    file.write(model_yaml)
model.save_weights('model.yaml.h5')
# 利用保存的网络结构复现一个一样的新的网络
with open('model.json', 'r') as file:  # 将保存的模型结构读入一个对象中
    new_model_json = file.read()
new_model = model_from_json(new_model_json)  # 利用模型结构对象构建新的网络结构
new_model.load_weights('model.json.h5')  # 利用保存的网络参数初始化网络
new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
new_model.fit(train_feature, train_flag_labels, epochs=200, batch_size=10)
new_score = new_model.evaluate(test_feature, test_flag_labels)
print(score)
print(new_score)