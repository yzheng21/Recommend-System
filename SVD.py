#coding=utf-8
from numpy import *
from numpy import linalg as la

#加载数据
def loadData():
    return mat([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])

#使用欧式距离计算物品之间的相似度
def ecludSim(inA,inB):
    return 1.0/(1.0+la.norm(inA-inB))

#选取svd分解后sigma矩阵中占全部矩阵信息90%的部分
def sigmak(sigma, pecentage):
    sigma2 = sigma**2
    sigma_2 = sum(sigma2)
    sigma_temp = 0
    k = 0
    for i in sigma:
        sigma_temp += i**2
        k += 1
        if sigma_temp >= sigma_2*pecentage:
            return k

#给用户未打分的item预估分数，根据与已打分item之间的相似度关系，得出预估分数
def scoreest(data, user_id, item, pecentage, sim=ecludSim):
    n = shape(data)[1]
    simtotal = 0.0
    ratetotal = 0.0
    u, sigma, vt = la.svd(data)
    k = sigmak(sigma, pecentage)
    sigma_matrix = mat(eye(k)*sigma[:k])     #得到sigma前k个主成分的对角矩阵
    V = data.T * u[:,:k] * sigma_matrix.I    #V n*k 是item在k维空间转换后的数据

    for j in range(n):
        userrating = data[user_id,j]
        if userrating == 0 or j == item:
            continue
        similarity = sim(V[item,:].T,V[j,:].T)
        simtotal += similarity
        ratetotal += similarity*userrating
    if similarity == 0:
        return 0
    else:
        return ratetotal/simtotal

#推荐预估分数前N个的物品
def recommend(data, user_id, N=2, sim = ecludSim, estmethod = scoreest, percentage = 0.9):
    unratingitem = nonzero(data[user_id,:].A == 0)[1]      #找到为评分物品的位置，一维数组
    if len(unratingitem) == 0:
        return 'rate every items'
    itemscore = []
    for item in unratingitem:
        score = estmethod(data, user_id, item , percentage)
        itemscore.append((item, score))
    itemscore = sorted(itemscore, key=lambda f: f[1], reverse=True)    #从高到低排序

    print(itemscore[:N])

data = loadData()
recommend(data,user_id=3)