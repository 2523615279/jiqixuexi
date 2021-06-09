from math import log
import operator

def create_data():
    dataSet = [
        ['sunny', 'hot', 'high', 'false', 'no'],
        ['sunny', 'hot', 'high', 'true', 'no'],
        ['overcast', 'hot', 'high', 'false', 'yes'],
        ['rainy', 'mild', 'high', 'false', 'yes'],
        ['rainy', 'cool', 'normal', 'false', 'yes'],
        ['rainy', 'cool', 'normal', 'true', 'no'],
        ['overcast', 'cool', 'normal', 'true', 'yes'],
        ['sunny', 'mild', 'high', 'false', 'no'],
        ['sunny', 'cool', 'normal', 'false', 'yes'],
        ['rainy', 'mild', 'normal', 'false', 'yes'],
        ['sunny', 'mild', 'normal', 'true', 'yes'],
        ['overcast', 'mild', 'high', 'true', 'yes'],
        ['overcast', 'hot', 'normal', 'false', 'yes'],
        ['rainy', 'mild', 'high', 'true', 'no']
            ]
    labels = ['outlook', 'temperature', 'humidity', 'windy', 'play']
    return dataSet, labels

def cal_entropy(dataSet):
    num = len(dataSet)
    label_count = {}
    for fea in dataSet:
        current_label = fea[-1]
        if current_label not in label_count.keys():
            label_count[current_label] = 0
        label_count[current_label] += 1
    entropy = 0.0
    for i in label_count:
        Pi = float(label_count[i]) / num
        entropy -= Pi * log(Pi, 2)
    return entropy

def remove_feature(dataSet, axis, feature):
    retdataset = []
    for featVec in dataSet:
        if featVec[axis] == feature:
            reducedata = featVec[:axis]
            reducedata.extend(featVec[axis + 1:])
            retdataset.append(reducedata)
    return retdataset

def choose_best_feature(dataSet):
    entropy = cal_entropy(dataSet)
    feature_num = len(dataSet[0]) - 1
    max_mutual_info = 0
    best_feature = -1
    for i in range(feature_num):
        feature_list = [example[i] for example in dataSet]
        feature_class = set(feature_list)
        conditional_entropy = 0
        for value in feature_class:
            retdataset = remove_feature(dataSet, i, value)
            Pi = len(retdataset) / float(len(dataSet))
            conditional_entropy += Pi * cal_entropy(retdataset)
        mutual_info = entropy - conditional_entropy
        if (mutual_info > max_mutual_info):
            max_mutual_info = mutual_info
            best_feature = i
    return best_feature

def majority_vote(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sort_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sort_class_count[0][0]

def create_tree(dataSet, labels):
    class_list = [example[-1] for example in dataSet]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataSet[0]) == 1:
        return majority_vote(class_list)
    best_feature = choose_best_feature(dataSet)
    best_feature_label = labels[best_feature]
    my_tree = {best_feature_label: {}}
    del (labels[best_feature])
    feature = [example[best_feature] for example in dataSet]
    feature_class = set(feature)
    for value in feature_class:
        sublabels = labels[:]
        my_tree[best_feature_label][value] = create_tree(remove_feature(dataSet, best_feature, value), sublabels)
    return my_tree

if __name__ == '__main__':
    dataSet, labels = create_data()
    print(create_tree(dataSet, labels))