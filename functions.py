import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.ensemble as ske
import threading
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import math
lock = threading.Lock()
def change_woe(d, cut, woe):
        list = []
        i = 0
        while i < len(d):
            value = d[i]
            j = len(cut) - 2
            m = len(cut) - 2
            while j >= 0:
                if value >= cut[j]:
                    j = -1
                else:
                    j -= 1
                    m -= 1
            list.append(woe[m])
            i += 1
        return list
def fill(data, datalist, listnotnull):
    for i in datalist:
        miss = data.iloc[:, [i, *listnotnull]]
        known = miss[data[data.columns[i]].notnull()].as_matrix()
        unknown = miss[data[data.columns[i]].isnull()].as_matrix()  # 存储为array
        train = known[:, 1:]
        result = known[:, 0]
        fil = ske.RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
        fil.fit(train, result)
        predict = fil.predict(unknown[:, 1:]).round(0)
        data.loc[data[data.columns[i]].isnull(), data.columns[i]] = predict
        listnotnull.append(i)
    return data

def change(data):
        for i in data.columns.tolist():
            select = data[i]
            Q1 = select.quantile(q=0.25)
            Q2 = select.quantile(q=0.75)
            Q = Q2 - Q1
            maxi = Q2 + 1.5 * Q
            mini = Q1 - 1.5 * Q
            mean = select.mean()
            data.loc[(data[i] > maxi), i] = mean
            data.loc[(data[i] < mini), i] = mean
            data.loc[(data[i] == float('inf')), i] = mean
        return data
def scorex(datac):
        for i in datac.columns.tolist():
            maxi = max(datac[i])
            mini = min(datac[i])
            rangei = maxi - mini
            datac[i] = (datac[i] - mini) / rangei * 100
        return datac
def rankx(data):
        rankdata = pd.DataFrame()
        rankdata[['A1', 'A2', 'A3', 'A4']] = data[['A1', 'A2', 'A3', 'A4']]
        for i in ['A1', 'A2', 'A3', 'A4']:
            Q1 = rankdata[i].quantile(q=0.2)
            Q2 = rankdata[i].quantile(q=0.4)
            Q3 = rankdata[i].quantile(q=0.6)
            Q4 = rankdata[i].quantile(q=0.8)
            rankdata.loc[(rankdata[i] < Q1), i] = 1
            rankdata.loc[(rankdata[i] < Q2) & (rankdata[i] >= Q1), i] = 2
            rankdata.loc[(rankdata[i] < Q3) & (rankdata[i] >= Q2), i] = 3
            rankdata.loc[(rankdata[i] < Q4) & (rankdata[i] >= Q3), i] = 4
            rankdata.loc[(rankdata[i] >= Q4), i] = 5
        rankdata[['偿债能力', '盈利能力', '成长能力', '运营能力']] = data[['A1', 'A2', 'A3', 'A4']]
        return rankdata

def pref(Path1,Path2,Q1,Q2):
    global lock
    lock.acquire()
    firstdata=pd.read_excel(Path1)
    seconddata=pd.read_excel(Path2)
    for i in firstdata['name']:
        if i not in seconddata['name'].tolist():
            firstdata = firstdata[~(firstdata['name'].isin([i]))]
    for i in seconddata['name']:
        if i not in firstdata['name'].tolist():
            seconddata = seconddata[~(seconddata['name'].isin([i]))]
    firstdata.reset_index(inplace=True)


    seconddata.reset_index(inplace=True)


    data = firstdata.iloc[:, 2:]
    data2 = seconddata.iloc[:, 2:]
    Q1.put(data)
    Q2.put(data2)
    lock.release()
def pre1(data,lo,Q,QQ):

    global lock

    lock.acquire()

    i = 0
    listnotnull = []
    while i < len(data.isnull().sum()):
        if data.isnull().sum()[i] == 0:
            listnotnull.append(i)
        i += 1

    datalist = list(range(len(data.columns)))



    for i in listnotnull:
        datalist.remove(i)








    data = fill(data, datalist, listnotnull)




    data['FIX'] = data['FA'] / data['TA']






    data = change(data)







    score1 = scorex(data)


    score1['A1'] = score1['CR'] * 0.5 + score1['DTAR'] * 0.5
    score1['A2'] = score1['ROTAR'] * 0.25 + score1['OPR'] * 0.25 + score1['RPCE'] * 0.25 + score1['ROI'] * 0.25
    score1['A3'] = score1['RAGR'] / 3 + score1['FAGR'] / 3 + score1['FIX'] / 3
    score1['A4'] = score1['RTR'] / 3 + score1['FAT'] / 3 + score1['TAR'] / 3






    rankdata1 = rankx(score1)



    rankdata1['score'] = (rankdata1['A1'] + rankdata1['A3']) * (rankdata1['A2'] + rankdata1['A4'])

    Q.put(rankdata1)
    QQ.put(data)

    lock.release()

def pre2(data2):
    global lock
    lock.acquire()







    i2 = 0
    listnotnull2 = []
    while i2 < len(data2.isnull().sum()):
        if data2.isnull().sum()[i2] == 0:
            listnotnull2.append(i2)

    datalist2 = list(range(len(data2.columns)))

    for i in listnotnull2:
        datalist2.remove(i)






    data2 = fill(data2, datalist2, listnotnull2)




    data2['FIX'] = data2['FA'] / data2['TA']







    data2 = change(data2)





    score2 = scorex(data2)


    score2['A1'] = score2['CR'] * 0.5 + score2['DTAR'] * 0.5
    score2['A2'] = score2['ROTAR'] * 0.25 + score2['OPR'] * 0.25 + score2['RPCE'] * 0.25 + score2['ROI'] * 0.25
    score2['A3'] = score2['RAGR'] / 3 + score2['FAGR'] / 3 + score2['FIX'] / 3
    score2['A4'] = score2['RTR'] / 3 + score2['FAT'] / 3 + score2['TAR'] / 3





    rankdata2 = rankx(data2)


    rankdata2['score'] = (rankdata2['A1'] + rankdata2['A3']) * (rankdata2['A2'] + rankdata2['A4'])



    lock.release()

def id(rankdata1,rankdata2,Q):
    global lock
    lock.acquire()
    x = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    y = []
    p1 = 0
    for i in range(9):
        y.append(sum((rankdata1['score'] >= p1) & (rankdata1['score'] < p1 + 10)))
        p1 += 10
    y.append(sum((rankdata1['score'] >= 90) & (rankdata1['score'] <= 100)))

    x2 = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    y2 = []
    p2 = 0
    for i in range(9):
        y2.append(sum((rankdata2['score'] >= p2) & (rankdata2['score'] < p2 + 10)))
        p2 += 10
    y2.append(sum((rankdata2['score'] >= 90) & (rankdata2['score'] <= 100)))



    plt.bar(x, y)
    plt.title('modelfile1')
    plt.xlabel('range')
    plt.ylabel('quantities')
    for a, b in zip(x, y):
        plt.text(a, b + 7, b, ha='center')
    plt.savefig(r'C:\Users\Administrator\Desktop\功能3结果\文件1各公司得分统计.png' )#模型文件1的分数分布图
    plt.show()


    plt.bar(x2, y2, color='r')
    plt.title('modelfile2')
    plt.xlabel('range')
    plt.ylabel('quantities')
    for a, b in zip(x2, y2):
        plt.text(a, b + 7, b, ha='center')
    plt.savefig(r'C:\Users\Administrator\Desktop\功能3结果\文件2各公司得分统计.png')#模型文件2的分数分布图
    plt.show()

    variety = pd.Series()
    variety = rankdata2['score'] - rankdata1['score']

    li = (variety > 0).tolist()
    lis = (variety <= 0).tolist()


    variety[li] = 0
    variety[lis] = 1
    Q.put(variety)
    lock.release()



def monoto_bin(Y, X, n=20):
    r = 0
    total_bad = Y.sum()
    total_good = Y.count() - total_bad
    while np.abs(r) < 1:
        #区间做index
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n, duplicates='drop')})
        d2 = d1.groupby('Bucket', as_index=True)
        #拆分，直到相关性为1(n=2时，只有两组，r=+—1，一定线性）,这样可以使得每个分组在一个区间内，差距不大
        #p需要判断吗
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    d3 = pd.DataFrame(d2.min().X, columns=['min_' + X.name])
    d3['min_' + X.name] = d2.min().X
    d3['max_' + X.name] = d2.max().X
    d3[Y.name] = d2.sum().Y
    d3['total'] = d2.count().Y
    # d3[Y.name + '_rate'] = d2.mean().Y
    #WOE公式
    d3['badattr'] = d3[Y.name] / total_bad
    d3['goodattr'] = (d3['total'] - d3[Y.name]) / total_good
    d3['woe'] = np.log(d3['goodattr'] / d3['badattr'])
    #iv公式
    iv = ((d3['goodattr'] - d3['badattr']) * d3['woe']).sum()
    d4 = (d3.sort_values(by='min_' + X.name)).reset_index(drop=True)
    cut = []
    cut.append(float('-inf'))
    for i in range(1, n + 1):
        qua = X.quantile(i / (n + 1))
        cut.append(round(qua, 4))
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4, iv, cut, woe

def change_score(series, cut, score):
        list = []
        i = 0
        while i < len(series):
            value = series[i]
            j = len(cut) - 2
            m = len(cut) - 2
            while j >= 0:
                if value >= cut[j]:
                    j = -1
                else:
                    j -= 1
                    m -= 1
            list.append(score[m])
            i += 1
        return list

def www(variety,data,q1,q2,q3,q4):
    global lock
    lock.acquire()
    dfx1, ivx1, cutx1, woex1 = monoto_bin(variety, data['OPR'], n=10)
    dfx2, ivx2, cutx2, woex2 = monoto_bin(variety, data['ROI'], n=10)
    dfx3, ivx3, cutx3, woex3 = monoto_bin(variety, data['ROTAR'], n=10)
    dfx4, ivx4, cutx4, woex4 = monoto_bin(variety, data['RPCE'], n=10)
    dfx5, ivx5, cutx5, woex5 = monoto_bin(variety, data['RTR'], n=10)
    dfx6, ivx6, cutx6, woex6 = monoto_bin(variety, data['TAR'], n=10)
    dfx7, ivx7, cutx7, woex7 = monoto_bin(variety, data['FAT'], n=10)
    dfx8, ivx8, cutx8, woex8 = monoto_bin(variety, data['DTAR'], n=10)
    dfx9, ivx9, cutx9, woex9 = monoto_bin(variety, data['CR'], n=10)
    dfx10, ivx10, cutx10, woex10 = monoto_bin(variety, data['RAGR'], n=10)
    dfx11, ivx11, cutx11, woex11 = monoto_bin(variety, data['FAGR'], n=10)
    dfx12, ivx12, cutx12, woex12 = monoto_bin(variety, data['FP'], n=10)
    dfx13, ivx13, cutx13, woex13 = monoto_bin(variety, data['TP'], n=10)
    dfx14, ivx14, cutx14, woex14 = monoto_bin(variety, data['FA'], n=10)
    dfx15, ivx15, cutx15, woex15 = monoto_bin(variety, data['TA'], n=10)
    dfx16, ivx16, cutx16, woex16 = monoto_bin(variety, data['FIX'], n=10)

    y = [ivx1, ivx2, ivx3, ivx4, ivx5, ivx6, ivx7, ivx8, ivx9, ivx10, ivx11, ivx12, ivx13, ivx14, ivx15, ivx16]


    index = data
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.bar(range(1, 17), y, width=0.4, color='r', alpha=0.6)  # 生成柱状图
    ax1.set_xticks(range(1, 17))
    ax1.set_xticklabels(index, rotation=0, fontsize=12)
    ax1.set_ylabel('iv', fontsize=14)
    # 在柱状图上添加数字标签
    for i, v in enumerate(y):
        plt.text(i + 1, v + 0.01, '%.4f' % v, ha='center', va='bottom', fontsize=12)
    plt.savefig(r'C:\Users\Administrator\Desktop\功能3结果\iv.png')#iv值图
    plt.show()


    def change_woe(d, cut, woe):
        list = []
        i = 0
        while i < len(d):
            value = d[i]
            j = len(cut) - 2
            m = len(cut) - 2
            while j >= 0:
                if value >= cut[j]:
                    j = -1
                else:
                    j -= 1
                    m -= 1
            list.append(woe[m])
            i += 1
        return list

    y = pd.Series(y)

    y1 = (y >= 0.01).tolist()


    datapre = data.iloc[:, y1]




    train = pd.DataFrame()
    cut = [cutx1, cutx2, cutx3, cutx4, cutx5, cutx6, cutx7, cutx8, cutx9, cutx10, cutx11, cutx12, cutx13, cutx14,
           cutx15, cutx16]
    woe = [woex1, woex2, woex3, woex4, woex5, woex6, woex7, woex8, woex9, woex10, woex11, woex12, woex13, woex14,
           woex15, woex16]




    for i in reversed(range(len(cut))):
        if y1[i] == False:
            cut.pop(i)
            woe.pop(i)


    for i, a, b in zip(datapre.columns.tolist(), cut, woe):
        train[i] = pd.Series(change_woe(datapre[i], a, b))




    train_x, test_x, train_y, test_y = train_test_split(train, variety, test_size=0.3, random_state=0)
    train1 = pd.concat([train_y, train_x], axis=1)
    test1 = pd.concat([test_y, test_x], axis=1)
    train1 = train1.reset_index(drop=True)
    test1 = test1.reset_index(drop=True)
    lr = LogisticRegression(penalty='l1')  # 如果模型的特征非常多，我们希望一些不重要的特征系数归零，从而让模型系数稀疏化的话，也可以使用L1正则化
    lr.fit(train_x, train_y)



    # y_pred= lr.predict(train_x)
   # train_predprob = lr.predict_proba(train_x)[:, 1]
    test_predprob = lr.predict_proba(test_x)[:, 1]
    FPR, TPR, threshold = roc_curve(test_y, test_predprob)
    ROC_AUC = auc(FPR, TPR)
    plt.plot(FPR, TPR, 'b', label='AUC = %0.2f' % ROC_AUC)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.savefig(r'C:\Users\Administrator\Desktop\功能3结果\auc.png')#AUC图
    plt.show()



    B = 20 / math.log(2)
    A = 600 - B / math.log(20)
    # 基础分
    base = round(A + B * lr.intercept_[0], 0)


    # 计算分数函数
    def compute_score(coe, woe, factor):
        scores = []
        for w in woe:
            score = round(coe * w * factor, 0)
            scores.append(score)
        return scores



    cscore = []
    for i, a in zip(range(len(cut)), woe):
        cscore.append(compute_score(lr.coef_[0][i], a, B))

    q1.put(train)
    q2.put(cut)
    q3.put(cscore)
    q4.put(base)

    lock.release()


def predict(train,cut,cscore,base,Path4):
    global lock
    lock.acquire()
    
    
    testdata = pd.read_excel(Path4)
    testdata1 = testdata.iloc[:, 1:]
    i = 0
    listnotnull = []
    while i < len(testdata1.isnull().sum()):
        if testdata1.isnull().sum()[i] == 0:
            listnotnull.append(i)
        i += 1

    datalist = list(range(len(testdata1.columns)))



    for i in listnotnull:
        datalist.remove(i)








    testdata1 = fill(testdata1, datalist, listnotnull)
    
    testdata1['FIX'] = testdata1['FA'] / testdata1['TA']


    testdata1 = change(testdata1)
    dd=scorex(testdata1)
    testdata1 = scorex(testdata1)
    
    dd['A1'] = dd['CR'] * 0.5 + dd['DTAR'] * 0.5
    dd['A2'] = dd['ROTAR'] * 0.25 + dd['OPR'] * 0.25 + dd['RPCE'] * 0.25 + dd['ROI'] * 0.25
    dd['A3'] = dd['RAGR'] / 3 + dd['FAGR'] / 3 + dd['FIX'] / 3
    dd['A4'] = dd['RTR'] / 3 + dd['FAT'] / 3 + dd['TAR'] / 3
    dd=rankx(dd)
    dd['总得分'] = (dd['A1'] + dd['A3']) * (dd['A2'] + dd['A4'])
    dd=dd.loc[:,['偿债能力', '盈利能力', '成长能力', '运营能力','总得分']]

    testdata2 = pd.DataFrame()

    for i, a, b in zip(train.columns.tolist(), cut, cscore):
        testdata2[i] = pd.Series(change_score(testdata1[i], a, b))
    

    testdata2['走势得分'] = base
    for i in testdata2.columns.tolist():
        testdata2['走势得分'] += testdata2[i]
    testdata2 -= base
    testdata2 = scorex(testdata2)
    testdata2['name']=testdata['name']
    testdata2=testdata2.loc[:,['走势得分','name']]
    testdata2['偿债能力']=dd['偿债能力']
    testdata2['成长能力']=dd['成长能力']
    testdata2['运营能力']=dd['运营能力']
    testdata2['盈利能力']=dd['盈利能力']
    testdata2['总得分']=dd['总得分']
    testdata2=testdata2.reindex(columns=['name','偿债能力', '盈利能力', '成长能力', '运营能力','总得分','走势得分'])
    testdata2.to_excel(r'C:\Users\Administrator\Desktop\功能3结果\预测文件变化趋势得分统计.xlsx')

    x3 = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    y3 = []
    p3 = 0
    for i in range(9):
        y3.append(sum((testdata2['走势得分'] >= p3) & (testdata2['走势得分'] < p3 + 10)))
        p3 += 10
    y3.append(sum((testdata2['走势得分'] >= 90) & (testdata2['走势得分'] <= 100)))

    plt.bar(x3, y3, color='g')
    plt.title('predictfile')
    plt.xlabel('scorerange')
    plt.ylabel('quantities')
    for a, b in zip(x3, y3):
        plt.text(a, b + 7, b, ha='center')
    plt.savefig(r'C:\Users\Administrator\Desktop\功能3结果\预测文件变化趋势得分统计图.png')#预测文件
    plt.show()
    lock.release()




def pre3(Path3,lo):

    global lock

    lock.acquire()
    firstdata=pd.read_excel(Path3)
    data = firstdata.iloc[:, 1:]
    i = 0
    listnotnull = []
    while i < len(data.isnull().sum()):
        if data.isnull().sum()[i] == 0:
            listnotnull.append(i)
        i += 1

    datalist = list(range(len(data.columns)))



    for i in listnotnull:
        datalist.remove(i)








    data = fill(data, datalist, listnotnull)




    data['FIX'] = data['FA'] / data['TA']






    data = change(data)







    score1 = scorex(data)


    score1['A1'] = score1['CR'] * 0.5 + score1['DTAR'] * 0.5
    score1['A2'] = score1['ROTAR'] * 0.25 + score1['OPR'] * 0.25 + score1['RPCE'] * 0.25 + score1['ROI'] * 0.25
    score1['A3'] = score1['RAGR'] / 3 + score1['FAGR'] / 3 + score1['FIX'] / 3
    score1['A4'] = score1['RTR'] / 3 + score1['FAT'] / 3 + score1['TAR'] / 3






    rankdata1 = rankx(score1)



    rankdata1['总得分'] = (rankdata1['A1'] + rankdata1['A3']) * (rankdata1['A2'] + rankdata1['A4'])
    rankdata1=rankdata1.loc[:,['偿债能力', '盈利能力', '成长能力', '运营能力','总得分']]
    rankdata1['name']=firstdata['name']
    rankdata1=rankdata1.reindex(columns=['name','偿债能力', '盈利能力', '成长能力', '运营能力','总得分'])
    rankdata1.to_excel(r'C:\Users\Administrator\Desktop\功能1结果\文件各公司得分情况.xlsx')

    x = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100']
    y = []
    p1 = 0
    for i in range(9):
        y.append(sum((rankdata1['总得分'] >= p1) & (rankdata1['总得分'] < p1 + 10)))
        p1 += 10
    y.append(sum((rankdata1['总得分'] >= 90) & (rankdata1['总得分'] <= 100)))



    plt.bar(x, y)
    plt.title('file')
    plt.xlabel('range')
    plt.ylabel('quantities')
    for a, b in zip(x, y):
        plt.text(a, b + 7, b, ha='center')
    plt.savefig(r'C:\Users\Administrator\Desktop\功能1结果\文件各公司得分分布统计图.png')  # 模型文件1的分数分布图
    plt.show()
    lock.release()

def radar(Path5,Firm):
    global lock

    lock.acquire()
    print(Firm)
    print(type(Firm))
    firstdata = pd.read_excel(Path5)
    data = firstdata.iloc[:, 1:]
    i = 0
    listnotnull = []
    while i < len(data.isnull().sum()):
        if data.isnull().sum()[i] == 0:
            listnotnull.append(i)
        i += 1

    datalist = list(range(len(data.columns)))

    for i in listnotnull:
        datalist.remove(i)

    data = fill(data, datalist, listnotnull)

    data['FIX'] = data['FA'] / data['TA']

    data = change(data)

    score1 = scorex(data)

    score1['A1'] = score1['CR'] * 0.5 + score1['DTAR'] * 0.5
    score1['A2'] = score1['ROTAR'] * 0.25 + score1['OPR'] * 0.25 + score1['RPCE'] * 0.25 + score1['ROI'] * 0.25
    score1['A3'] = score1['RAGR'] / 3 + score1['FAGR'] / 3 + score1['FIX'] / 3
    score1['A4'] = score1['RTR'] / 3 + score1['FAT'] / 3 + score1['TAR'] / 3

    rankdata1 = rankx(score1)
    rankdata1 = rankdata1.loc[:, ['偿债能力', '盈利能力', '成长能力', '运营能力']]
    rankdata1['name'] = firstdata['name']
    rankdata1 = rankdata1.reindex(columns=['name', '偿债能力', '盈利能力', '成长能力', '运营能力'])
    firm=rankdata1.iloc[rankdata1[rankdata1['name']==Firm.strip()].index.tolist()[0],:]

    matplotlib.rcParams['font.family'] = 'SimHei'
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    labels = np.array(['偿债能力', '盈利能力', '成长能力', '运营能力'])
    nAttr = 4
    Python = np.array([firm['偿债能力'],firm['盈利能力'], firm['成长能力'], firm['运营能力']])
    angles = np.linspace(0, 2 * np.pi, nAttr, endpoint=False)
    Python = np.concatenate((Python, [Python[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    fig = plt.figure(facecolor="white")
    a = fig.add_subplot(111, polar=True)
    a.plot(angles, Python, 'bo-', color='g', linewidth=2)
    a.set_title(Firm.strip()+'能力图',x=0,y=1)
    a.set_rlim(0, 100)
    a.fill(angles, Python, facecolor='g', alpha=0.2)
    a.set_thetagrids(angles * 180 / np.pi, labels)
    a.grid(True)
    plt.savefig(r'C:\Users\Administrator\Desktop\功能2结果\雷达图.png')
    plt.show()
    lock.release()