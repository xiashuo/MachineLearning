# _*_ coding:utf-8 _*_
from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1代表侮辱性文字，0代表正常言论
    # 返回的第一个变量是进行词条切分后的文档集合，这些文档来自斑点犬爱好者留言板。
    #返回的第二个变量是一个类别标签的集合。这里有两类，侮辱性和非侮辱性。
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])  #创建一个空集合
    for document in dataSet:
        vocabSet = vocabSet | set(document) #操作符|用于求两个集合的并集
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet): #参数inputSet为词汇表及某个文档
    returnVec = [0]*len(vocabList) #创建一个和词汇表等长的向量，并将其元素都设置为0
    for word in inputSet:   #遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec  #返回的是文档向量，向量的每一元素为1或0

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1 #与之前唯一不同是，每当遇到一个单词时，增加词向量中的对应值，而不只是将对应的数值设为1。
    return returnVec

def trainNB0(trainMatrix,trainCategory): #参数trainMatrix是转换成数字后的文档矩阵
    numTrainDocs = len(trainMatrix) #文档的数量
    numWords = len(trainMatrix[0]) #每一行的元素个数，这里就等于词汇表里单词的个数
    pAbusive = sum(trainCategory)/float(numTrainDocs) #类别1的概率，即p(1)
    # p0Num = zeros(numWords); p1Num = zeros(numWords)
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones()
    # p0Denom = 0.0; p1Denom = 0.0
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):  #遍历所有文档
        if trainCategory[i] == 1: # 如果该篇文档的类别为1
            p1Num += trainMatrix[i] #则对应分类中，所有出现过的单词都加1
            p1Denom += sum(trainMatrix[i]) #将该文档中单词的总数累加
        else:  #如果是类别0，则相反执行相同的操作
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #计算每个单词属于类别1的概率
    # p1Vect = p1Num/p1Denom          #change to log()
    p1Vect = log(p1Num/p1Denom)          #change to log()
    #计算没个单词属于类别0的概率
    # p0Vect = p0Num/p0Denom          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1): #参数vec2Classify是转换为数字的文档
    # 这里用到了上面的贝叶斯公式，因为用log函数处理后，乘法变成了加法。
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #计算该文档属于类别1的概率
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1) #计算该文档属于类别0的概率
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    print(myVocabList)
    trainMat=[]
    trainMat = [setOfWords2Vec(myVocabList, postinDoc) for postinDoc in listOPosts]
    p0v,p1v,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0v, p1v, pAb))
    testEntry = ['him', 'so', 'stupid']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0v, p1v, pAb))


def textParse(bigString):  # 输入参数为一串很长的字符串
    import re
    listOfTokens = re.split(r'\W*', bigString) #利用正则表达式，split函数进行切分，分隔符是除单词、数字外的任意字符串。
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] #全部转换为小写，并返回得到的一系列词组成的词表，只返回长度大于2的词

def spamTest():
    docList = []; #创建所有邮件单词集合，初始为空
    classList = [];#标签集合
    for i in range(1, 26): #文件里一共有25封邮件，遍历25次
        #这里读取文档时会存在编码问题，用如下方式解决
        wordList = textParse(open('email/spam/%d.txt' % i,"rb").read().decode('GBK','ignore'))
        docList.append(wordList) #将该篇邮件的词集加入到总词集中
        classList.append(1) #添加标签1
        wordList = textParse(open('email/ham/%d.txt' % i, "rb").read().decode('GBK', 'ignore'))
        docList.append(wordList)  #将该篇邮件的词集加入到总词集中
        classList.append(0) #添加标签0
    vocabList = createVocabList(docList)  # 生成单词集合（去掉重复的）
    trainingSet = list(range(50)); #创建一个大小为50的list
    testSet = []  # 测试集
    for i in range(10): #随机选择10篇邮件的向量词集作为测试样本
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex]) #删除掉这10组，剩下的40组作为训练样本
    trainMat = []; #训练集合矩阵
    trainClasses = [] #训练标签集
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))#将每个训练样本单词向量加入到训练矩阵中
        trainClasses.append(classList[docIndex]) #将每个训练样本的标签加入到训练标签集合
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses)) #计算属于对应标签的概率
    errorCount = 0
    #测试10组测试样本的预测结果，并打印错误率
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    error_rate=float(errorCount) / len(testSet)
    print('the error rate is: ',error_rate)
    return error_rate
    # return vocabList,fullText


if __name__=="__main__":
    errorRate=0
    for i in range(10):
        errorRate +=spamTest()
    print("平均错误率为",float(errorRate/10))


    # import feedparser
    # ny=feedparser.parse('')

    # listPost,listClasses=loadDataSet()
    # myVocabList=createVocabList(listPost)
    # print(myVocabList)
    # # print(setOfWords2Vec(myVocabList,listPost[0]))
    # # print(setOfWords2Vec(myVocabList,listPost[1]))
    # trainMat=[setOfWords2Vec(myVocabList,post) for post in listPost]
    # p0v,p1v,pAb=trainNB0(trainMat,listClasses)
    # print(pAb)
    # print(p0v)
    # print(p1v)
    # testingNB()



