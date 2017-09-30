# _*_ coding:utf-8 _*_
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):                      #next 2 lines create weights matrix
        diffMat = testPoint - xMat[j,:]     #
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):  #loops over all the data points and applies lwlr to each one
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def lwlrTestPlot(xArr,yArr,k=1.0):  #same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))       #easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy

if __name__=="__main__":
    xArr,yArr=loadDataSet("ex0.txt")
    # w=standRegres(xArr,yArr)
    xMat=mat(xArr)
    yMat=mat(yArr)
    # yHat=xArr*w
    yHat1,xCopy=lwlrTestPlot(xArr,yArr)
    yHat2,xCopy=lwlrTestPlot(xArr,yArr,0.01)
    yHat3,xCopy=lwlrTestPlot(xArr,yArr,0.003)
    fig=plt.figure()
    ax1=fig.add_subplot(221)
    ax1.scatter(xMat[:,1],yMat)
    ax1.plot(xCopy[:,1],yHat1)
    ax2=fig.add_subplot(222)
    ax2.scatter(xMat[:,1],yMat)
    ax2.plot(xCopy[:,1],yHat2)
    ax3=fig.add_subplot(223)
    ax3.scatter(xMat[:,1],yMat)
    ax3.plot(xCopy[:,1],yHat3)
    plt.show()

