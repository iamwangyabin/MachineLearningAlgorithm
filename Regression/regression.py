from numpy import *

#该文件由三段数字组成，中间用tab分割
def loadDataSet(filename):
    #把一行以tab分解成列表
    numFeat=len(open(filename).readline().split('\t'))-1
    dataMat=[]
    labelMat=[]
    fr=open(filename)

    for line in fr.readlines():
        lineArr=[]
        #strip是去换行，split是分割成列表
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#计算最佳拟合曲线，就是一个公式的实现
def standRegres(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xTx=xMat.T*xMat
    if linalg.det(xTx)==0.0:
        print("This matrix is singular ,cannot do inverse")
        return
    ws=xTx.I*(xMat.T*yMat)
    return ws

def start():
    xArr,yArr=loadDataSet('/home/wang/PycharmProjects/MachineLearningAlgorithm/Regression/ex0.txt')
    ws=standRegres(xArr,yArr)
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_subplot(111)
    xMat=mat(xArr)
    yMat=mat(yArr)
    # flatten是把多维数据降成一维
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
    xCopy=xMat.copy()
    xCopy.sort(0)
    yHat=xCopy*ws
    ax.plot(xCopy[:,1],yHat)
    plt.show()

def lwlr(testPoint,xArr,yArr,k=1.0):
    pass