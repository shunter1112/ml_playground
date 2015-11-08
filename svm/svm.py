#coding:utf-8

import numpy as np
from pylab import *

LR = 0.05    # 学習率
C = sys.maxint    # Cを無限大に設定
CountMax = 1000   # 1000回更新

def f(x1, w, b):
    return - (w[0] / w[1]) * x1 - (b / w[1])

def kernel(x, y):
    return np.dot(x, y)  # 線形カーネル

def dL(i):
    ans = 0
    for j in range(0,N):
        ans += L[j] * t[i] * t[j] * kernel(X[i], X[j])
    return (1 - ans)

if __name__ == "__main__":
    data = np.loadtxt("data/svm_hard_data.txt")
    N = data.size / 3                   # 教師データ数
    X = np.delete(data,2,1)             # 入力の座標
    X = np.c_[X,np.ones(X.shape[0])]    # 入力の空間を1次元拡張
    t = np.array(data[:,2])             # 入力のラベル
    L = np.zeros((N,1))                 # データの個数分のラグランジュ乗数

    count = 0
    while (count < CountMax):
        for i in range(N):
            L[i] = L[i] + LR * dL(i)    # ラグランジュ乗数の更新
            if (L[i] < 0):
                L[i] = 0
            elif (L[i] > C):
                L[i] = C
        count += 1

    for i in range(N):
        print L[i]

    # サポートベクトルのインデックスを抽出
    S = []
    for i in range(len(L)):
        if L[i] < 0.00001: continue
        S.append(i)

    # wを計算
    w = np.zeros(3)
    for n in S:
        w += L[n] * t[n] * X[n]

    # wの3次元目は拡張次元のbとなる。
    b = w[2]
    np.delete(w, 2, 0)

    # 訓練データを描画
    for i in range(0,N):
        if(t[i] > 0):
            plot(X[i][0],X[i][1], 'rx')
        else:
            plot(X[i][0],X[i][1], 'bx')

    # サポートベクトルを描画
    for n in S:
        scatter(X[n,0], X[n,1], s=80, c='y', marker='o')

    # 識別境界を描画
    x1 = np.linspace(-6, 6, 1000)
    x2 = [f(x, w, b) for x in x1]
    plot(x1, x2, 'g-')

    xlim(-6, 6)
    ylim(-6, 6)
    show()