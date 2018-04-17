from numpy import *


def loadSimpData():
    dataMat = matrix([[1.0, 2.1],
                      [2.0, 1.1],
                      [1.3, 1.0],
                      [1.0, 1.0],
                      [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels
#thresh反复做,liI
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    #返回矩阵是一个n维列向量,对数据集每一列的各个特征进行阈值过滤
    retArray=ones((shape(dataMatrix)[0],1))
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix=mat(dataArr)
    #转置完成后是一个行向量
    labelMat=mat(classLabels).T
    m,n=shape(dataMatrix)
    numSteps=10.0
    bestStump={}
    #最优单层决策树的预测结果
    bestClassEst=mat(zeros((m,1)))
    minError=inf
    #第一个循环遍历所有特征
    for i in range(n):
        #这个函数每次取第i列的最小值和最大值,返回是个float。
        rangeMin=dataMatrix[:,i].min()
        rangeMax=dataMatrix[:,i].max()
        stepSize=(rangeMax-rangeMin)/numSteps
        #遍历每个步长
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal=(rangeMin+float(j)*stepSize)
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr=mat(ones((m,1)))
                errArr[predictedVals==labelMat]=0
                weightedError=D.T*errArr
                print("split: dim %d, thresh %.2f,thresh inequal: %s,the weighted error is %.3f" % (i,threshVal,inequal,weightedError))
                if weightedError<minError:
                    minError=weightedError
                    bestClassEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClassEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr=[]
    m=shape(dataArr)[0]
    D=mat(ones((m,1))/m)
    aggClassEst=mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        print("D:",D.T)
        alpha=float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha']=alpha
        weakClassArr.append(bestStump)
        print("classEst: ",classEst.T)
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        D=multiply(D,exp(expon))
        D=D/D.sum()
        aggClassEst+=alpha*classEst
        print("aggClassEst: ",aggClassEst.T)
        aggErrors=multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate=aggErrors.sum()/m
        print("total error: ",errorRate,"\n")
        if errorRate==0.0:
            break
    return weakClassArr

def adaClassify(dataToClass,classifierArr):
    dataMatrix=mat(dataToClass)
    m=shape(dataMatrix)[0]
    aggClassEst=mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)