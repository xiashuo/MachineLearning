# _*_ coding:utf-8 _*_
from math import log
from copy import deepcopy
import treePlotter
import operator
def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #计算数据集中实例的总数
    labelCounts = {}   #用来记录每个标签出现的次数
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1] #每个实例最后一个特征元素为其分类标签
        #如果当前标签在labelCounts.keys()中存在，则该标签计数加1
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries #这个是每个标签出现的概率，相当于公式里的P(Xi)
        shannonEnt -= prob * log(prob,2) #利用了上面的熵计算公式
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def splitDataSet(dataSet, axis, value):#dataSet为要划分的数据集，axis是以哪个特征划分，value是该特征的值
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  #把在axis左边的元素放在列表里
            reducedFeatVec.extend(featVec[axis+1:])#把在axis右边的元素放在列表里
            retDataSet.append(reducedFeatVec) #将上面修改后的列表加入到划分数据集中
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #每行最后一列是标签，故减1
    baseEntropy = calcShannonEnt(dataSet) #计算初始熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #遍历所有的特征划分
        featList = [example[i] for example in dataSet]#将每行数据的第i列存入列表中
        # print(featList)
        uniqueVals = set(featList)   #列表会有相同的数据，使用set去掉重复的
        # print(uniqueVals)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) #利用第i个特征进行划分，并对每个唯一特征值求得的熵求和
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy     #计算信息增益
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain         #选信息增益最大的特征的索引，并返回
            bestFeature = i
    return bestFeature

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet] #获取数据集的所有类标签
    if classList.count(classList[0]) == len(classList): #如果所有类标签完全相同，则返回该类标签，递归结束
        return classList[0]
    if len(dataSet[0]) == 1: #如果所有的特征已经使用完了，仍然没有是数据集划分成仅包含唯一类别的分组，
        return majorityCnt(classList)#则返回出现次数最多的标签
    bestFeat = chooseBestFeatureToSplit(dataSet)#选择最好的划分特征，这里返回的是最好特征的索引值
    bestFeatLabel = labels[bestFeat] #最好特征的值
    myTree = {bestFeatLabel:{}} #初始化树结构
    del(labels[bestFeat]) #删除这个特征
    featValues = [example[bestFeat] for example in dataSet] #获取该特征的所有取值
    uniqueVals = set(featValues) #用set函数，去掉相同重复的
    for value in uniqueVals:
        subLabels = labels[:] #在Python语言中函数参数是列表类型时，参数是按照引用方式传递的，为了保证每次调用函数createTree()时不改变原始列表的内容，使用新变量subLabels代替原始列表。
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)#进行递归构建，重复上面的过程
    return myTree

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        # if vote not in classCount.keys(): classCount[vote] = 0
        # classCount[vote] += 1
        classCount[vote]=classCount.get(vote,0)+1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree)[0] #当前节点的标签名称
    secondDict = inputTree[firstStr] #子树
    featIndex = featLabels.index(firstStr)# 将标签转换为索引
    key = testVec[featIndex] #当前的特征标签
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):#判断是否为字典类型
        classLabel = classify(valueOfFeat, featLabels, testVec) #如果是，则继续递归分类
    else: classLabel = valueOfFeat #否则，返回当前标签
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'ab')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,"rb").read().decode('GBK','ignore')
    return pickle.load(fr)

def testShang():
    myDat, labels = createDataSet()
    print(myDat)
    print(calcShannonEnt(myDat))
def testsplitDataSet():
    myDat, labels = createDataSet()
    print(myDat)
    print(splitDataSet(myDat,0,0))
    print(splitDataSet(myDat,0,1))
    print(splitDataSet(myDat,1,0))
    print(splitDataSet(myDat,1,1))
def testschooseBestFeatureToSplit():
    myDat, labels = createDataSet()
    print(myDat)
    print(chooseBestFeatureToSplit(myDat))


if __name__=="__main__":
    # myDat,labels=createDataSet()
    # # featLabels=deepcopy(labels)
    # myTree=createTree(myDat, labels)
    # print(myTree)
    # storeTree(myTree,"classifierStorage.txt")
    # newTree=grabTree("classifierStorage.txt")
    # print(newTree)
    # print(classify(myTree,featLabels,[1,1]))
    # print(classify(myTree,featLabels,[0,0]))
    fr=open("lenses.txt")
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]
    labels=['age','prescript','astigmatic','tearrate']
    lensesTree=createTree(lenses,labels)
    treePlotter.createPlot(lensesTree)
    # testShang()
    # testsplitDataSet()
    # testschooseBestFeatureToSplit()

