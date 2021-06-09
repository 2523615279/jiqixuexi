from math import log

def createDataSet():
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

def calcShannonEnt(dataSet):
    numEntires = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntires
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

if __name__ == '__main__':
    dataSet, features = createDataSet()
    print(dataSet)
    print(calcShannonEnt(dataSet))
