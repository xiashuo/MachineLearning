# _*_ coding:utf-8 _*_
from numpy import *
import operator
def createDataSet():
    dataSet=array([[1,1],[1,0],[3,3],[3,4]])
    labels=['A','A','B','B']
    return dataSet,labels

def knn_classify(inputX,dataSet,labels,k):
    dataSize=dataSet.shape[0]
    a=tile(inputX,(dataSize,1))-dataSet
    b=(a**2).sum(axis=1)
    distances=b**0.5
    # print(distances)
    sortedIndex=distances.argsort()
    # print(sortedIndex)
    classCount={}
    for i in range(k):
        label=labels[sortedIndex[i]]
        # print(sortedIndex[i])
        # print(label)
        classCount[label]=classCount.get(label,0)+1
    maxCount=0
    # print(classCount)
    # for key,value in classCount.items():
    #     if value >maxCount:
    #         maxCount=value
    #         output_label=key
    # return output_label
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=1)
    return sortedClassCount[0][0]

if __name__=="__main__":
    data_set,labels=createDataSet()
    input=array([3,3.8])
    output_label=knn_classify(input,data_set,labels,3)
    print("标签为：",output_label)


