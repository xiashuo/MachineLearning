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

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()


def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print
        "This matrix is singular, cannot do inverse"
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # to eliminate X0 take mean off of Y
    # regularize X's
    xMeans = mean(xMat, 0)  # calc mean then subtract it off
    xVar = var(xMat, 0)  # calc variance of Xi then divide by it
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

if __name__=="__main__":
    # xArr,yArr=loadDataSet("ex0.txt")
    # # w=standRegres(xArr,yArr)
    # xMat=mat(xArr)
    # yMat=mat(yArr)
    # # yHat=xArr*w
    # yHat1,xCopy=lwlrTestPlot(xArr,yArr)
    # yHat2,xCopy=lwlrTestPlot(xArr,yArr,0.01)
    # yHat3,xCopy=lwlrTestPlot(xArr,yArr,0.003)
    # fig=plt.figure()
    # ax1=fig.add_subplot(221)
    # ax1.scatter(xMat[:,1],yMat)
    # ax1.plot(xCopy[:,1],yHat1)
    # ax2=fig.add_subplot(222)
    # ax2.scatter(xMat[:,1],yMat)
    # ax2.plot(xCopy[:,1],yHat2)
    # ax3=fig.add_subplot(223)
    # ax3.scatter(xMat[:,1],yMat)
    # ax3.plot(xCopy[:,1],yHat3)
    # plt.show()
    xArr, yArr = loadDataSet("abalone.txt")
    # yHat1=lwlrTest(xArr[100:199],xArr[0:99],yArr[0:99])
    # yHat2=lwlrTest(xArr[100:199],xArr[0:99],yArr[0:99],0.1)
    # yHat3=lwlrTest(xArr[100:199],xArr[0:99],yArr[0:99],10)
    # print(rssError(yArr[100:199],yHat1.T))
    # print(rssError(yArr[100:199],yHat2.T))
    # print(rssError(yArr[100:199],yHat3.T))
    rweights=ridgeTest(xArr,yArr)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(rweights)
    plt.show()


