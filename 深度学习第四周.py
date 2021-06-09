import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

dataset = pd.read_csv('bank.csv', delimiter=';')

dataset['job'] = dataset['job'].replace(to_replace=["admin.", "unknown", "unemployed", "management",
                                                    "housemaid", "entrepreneur", "student",
                                                    "blue-collar", "self-employed", "retired", "technician",
                                                    "services"],
                                        value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
dataset['marital'] = dataset['marital'].replace(to_replace=["married", "divorced", "single"], value=[0, 1, 2])
dataset['education'] = dataset['education'].replace(to_replace=["unknown", "secondary", "primary", "tertiary"],
                                                    value=[0, 1, 2, 3])
dataset['default'] = dataset['default'].replace(to_replace=["yes", "no"], value=[0, 1])
dataset['housing'] = dataset['housing'].replace(to_replace=["yes", "no"], value=[0, 1])
dataset['loan'] = dataset['loan'].replace(to_replace=["yes", "no"], value=[0, 1])
dataset['contact'] = dataset['contact'].replace(to_replace={"unknown", "telephone", "cellular"}, value=[0, 1, 2])
dataset['month'] = dataset['month'].replace(to_replace={"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
                                                        "oct", "nov", "dec"},
                                            value=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
dataset['poutcome'] = dataset['poutcome'].replace(to_replace={"unknown", "other", "failure", "success"},
                                                  value=[0, 1, 2, 3])
dataset['y'] = dataset['y'].replace(to_replace=["yes", "no"], value=[0, 1])
# dataset_new = np.array(dataset)
dataset_new = dataset.values
feature = dataset_new[:, 0:16]
flag = dataset_new[:, 16]
# data_num = len(feature)
# train_num = int(data_num * 0.8)  # 求训练集的个数
#
# feature_train = feature[:train_num, :]  # 取前train_num行作为训练集特征
# feature_test = feature[train_num:, :]  # 取后train_num行作为训练集特征
# flag_train = flag[:train_num]
# flag_test = flag[train_num:]
train_feature, test_feature, train_flag, test_flag = train_test_split(feature, flag, train_size=0.2)


def create_model():
    model = Sequential()
    model.add(Dense(units=12, input_dim=16, activation='relu', kernel_initializer='normal'))
    model.add(Dense(units=8, activation='relu', kernel_initializer='normal'))
    model.add(Dense(units=4, activation='relu', kernel_initializer='normal'))
    model.add(Dense(units=1, activation='relu', kernel_initializer='normal'))
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=10)
k = KFold(n_splits=3, shuffle=True)
result = cross_val_score(model, feature, flag, cv=k)
print(result)
