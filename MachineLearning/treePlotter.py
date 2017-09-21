# _*_ coding:utf-8 _*_
import matplotlib.pyplot as plt
# 定义决策树决策结果的属性，用字典来定义
# 下面的字典定义也可写作 decisionNode={boxstyle:'sawtooth',fc:'0.8'}
# boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree)[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 判断节点值的类型是否为字典
            numLeafs += getNumLeafs(secondDict[key]) #如果是，则进行递归，进入下个节点
        else:       #如果不是，则该节点为叶子节点。
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree)[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict': #判断节点值的类型是否为字典
            thisDepth = 1 + getTreeDepth(secondDict[key]) #如果是，则当前深度加1，并递归进入下个节点。
        else:
            thisDepth = 1  #如果是，则当前深度为1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth #循环完毕，返回最大深度值


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

#在两个节点中间标注
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):  # if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  # 当前树的叶子节点数目
    depth = getTreeDepth(myTree) #这里好像并没有用到
    firstStr = list(myTree)[0]  # 当前第一个节点
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)#确定当前节点的绘制位置
    plotMidText(cntrPt, parentPt, nodeTxt) #标注两个节点之间的信息
    plotNode(firstStr, cntrPt, parentPt, decisionNode)#绘制决策节点
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key], cntrPt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# if you do get a dictonary you know it's a tree, and the first element will be another dict

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white') # 定义一个画布，背景为白色
    fig.clf() # 把画布清空
    axprops = dict(xticks=[], yticks=[])# 定义横纵坐标轴，无内容
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)# 绘制图像,无边框,无坐标轴
    plotTree.totalW = float(getNumLeafs(inTree))  #全局变量宽度 = 叶子数
    plotTree.totalD = float(getTreeDepth(inTree)) #全局变量高度 = 深度
    plotTree.xOff = -0.5 / plotTree.totalW; #例如绘制3个叶子结点，坐标应为1/3,2/3,3/3，为了显示效果，将x向左移一点（我只能这么理解了）
    plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


# def createPlot():
#    fig = plt.figure(1, facecolor='white')
#    fig.clf()
#    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
#    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#    plt.show()

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]
def test():
    myTree = retrieveTree(0)
    print(getNumLeafs(myTree))
    print(getTreeDepth(myTree))

if __name__=="__main__":
    myTree=retrieveTree(0)
    print(myTree)
    createPlot(myTree)
    # myTree['no surfacing'][3]='maybe'
    # print(myTree['no surfacing'])
    # createPlot(myTree)
    # test()