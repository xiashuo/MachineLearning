# _*_ coding:utf-8 _*_

from numpy import *
import matplotlib.pyplot as plt

#便利函数，主要功能是打开文本文件testSet.txt并逐行读取。
def loadDataSet():
    dataMat = []; labelMat = [] #数据集和标签集
    fr = open('testSet.txt') #该文件每行前两个值分别是X1和X2，第三个值是数据对应的类别标签。
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #为了方便计算，这里将X0的值设为1.0。
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)             #转换成numpy矩阵
    labelMat = mat(classLabels).transpose() #转换成numpy矩阵，并转置
    m,n = shape(dataMatrix)   #获取行值m和列的值n
    alpha = 0.001            #变量alpha是向目标移动的步长
    maxCycles = 500          #maxCycles是迭代次数
    weights = ones((n,1))   #初始化weights，n行，1列，全为1
    for k in range(maxCycles):              #迭代
        h = sigmoid(dataMatrix*weights) #h是m行，1列的矩阵
        # print(h)
        error = (labelMat - h)              #计算真实类别与预测类别的差值
        # print(error)
        weights = weights + alpha * dataMatrix.transpose()* error #按照该差值的方向调整回归系数
        # print(weights)
    return weights,h     #返回训练好的回归系数

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights)) #这里为数值
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=200):#第三个参数是迭代次数，默认200次
    m,n = shape(dataMatrix)
    weights = ones(n)   #initialize to all ones
    for j in range(numIter): #迭代
        dataIndex = list(range(m))#返回的是[0,1,2,...m-1,m]
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #alpha在每次迭代的时候都会调整
            randIndex = int(random.uniform(0,len(dataIndex)))#这里通过随机选取样本来更新回归系数。
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def plotBestFit(weights):
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = [] #分类为1的点
    xcord2 = []; ycord2 = [] #分类为0的点
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1) #从-3.0到3.0，间隔为0.1，即[-3.0,-2.9,-2.8,...2.9,3.0]
    y = (-weights[0]-weights[1]*x)/weights[2] #这里就是W[0]*1+W[1]*X+W[2]*Y=0的变形，等于0就是sigmoid函数为0，0为分界点。
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21): #将每个样本的前20个特征和最后一个标签分开存储到相应矩阵
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500) #利用随机梯度上升函数计算回归系数向量，迭代500次
    errorCount = 0; numTestVec = 0.0 #错误次数和测试样本数量
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate
#调用函数colicTest()10次并求结果的平均值。
def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

if __name__=="__main__":
    dataArr,labelMat=loadDataSet()
    weights,h=gradAscent(dataArr,labelMat)
    print(weights)
    print(h)
    # plotBestFit(weights.getA()) #将weight转换成numpy.array形式
    # weights=stocGradAscent1(array(dataArr),labelMat) #这里需要将dataArr转换成numpy.array形式
    # plotBestFit(weights)
    # multiTest()