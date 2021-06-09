from sklearn import datasets
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json

# 数据切片
dataset = datasets.load_iris()
feature = dataset.data
flag = dataset.target

# 划分测试集，训练集
train_feature, test_feature, train_flag, test_flag = train_test_split(feature, flag, train_size=0.2)
# 转换成独热编码
train_flag_labels = to_categorical(train_flag, num_classes=3)
test_flag_labels = to_categorical(test_flag, num_classes=3)


def create_model():
    """
    定义函数create_model
    实现模型的网络结构
    :return: model
    """
    model = Sequential()
    model.add(Dense(6, input_dim=4, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(12, activation='tanh', kernel_initializer='glorot_uniform'))
    model.add(Dense(8, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(3, activation='softmax', kernel_initializer='glorot_uniform'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def load_model():
    """
    定义函数load_model
    实现复现模型结构
    复现模型参数
    :return: new_model
    """
    with open('model.json', 'r') as file:
        new_model_json = file.read()
    new_model = model_from_json(new_model_json)
    new_model.load_weights('./test/weight.h5')
    new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return new_model


# file = 'D:/jiqixuexi/test/weight-{epoch:02d}-{accuracy:2f}.h5'
file = './test/weight.h5' # 存放模型参数的地址
checkpoint = ModelCheckpoint(filepath=file, monitor='accuracy', save_best_only=True,
                             save_weights_only=True) # 取训练过程中精度最高的
checkpoint_list = [checkpoint]

model = create_model()
model.fit(train_feature, train_flag_labels, epochs=200, batch_size=10, callbacks=checkpoint_list)   # 训练模型

# 以json格式保存网络模型
# 保存网络结构
model_json = model.to_json()
with open('model.json', 'w') as file:
    file.write(model_json)

new_model = load_model()
new_model.fit(train_feature, train_flag_labels, epochs=200, batch_size=10)

new_score = new_model.evaluate(test_feature, test_flag_labels) # 测试模型
score = model.evaluate(test_feature, test_flag_labels)
print(score)
print(new_score)
