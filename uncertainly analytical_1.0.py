import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import time
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
k_list = genfromtxt('k_values.txt')
k_list = np.sqrt(k_list)
print(k_list)
start = time.time()
max_Temp = []
for K in k_list:
    b = 0.012
    a = 0.012
    m = 15000/K
    gamma = 2.9
    T0 = 341.8

    Power = genfromtxt('Chiplet_Core0_Power0_1.ptrace')
    FUnit = genfromtxt('Chiplet_Core0_1.flp')
    num_power = Power.shape[1]  # 读取矩阵的维度                  #112
    y_cor = set()  # 创建一个无序不重复元素集y_cor
    x_cor = set()
    for i in range(num_power):
        y_cor.add(FUnit[i, 4])  # y_cor为提出F中第五列的所有不重复元素并乱序排列
        x_cor.add(FUnit[i, 3])
    y_cor.add(a)
    x_cor.add(b)
    y_cor = list(y_cor)
    x_cor = list(x_cor)
    y_cor.sort()
    x_cor.sort()


    def f(x, y):
        for i in range(num_power):
            if (x >= FUnit[i, 3] and x <= FUnit[i, 3] + FUnit[i, 1] and y >= FUnit[i, 4] and y <= FUnit[i, 4] + FUnit[
                i, 2]):
                return -Power[1, i] / (FUnit[i, 1] * FUnit[i, 2] * 0.00015) / K/K / gamma - m * m * T0



    num_eigen = 20  # Cpq为20*20矩阵
    fy = {}  # 将对fxy的x积分后的表达式
    num_seg_y = len(y_cor) - 1  # 剔除a后的长度:18
    num_seg_x = len(x_cor) - 1
    fy[0] = np.zeros((num_seg_y,))  # 创建指定大小的数组，数组元素以 0 来填充,将其给键值fy[0]

    for i in range(num_seg_y):
        temp = 0  # 积分累积变量

        y = (y_cor[i] + y_cor[i + 1]) / 2
        for j in range(num_seg_x):
            x = (x_cor[j + 1] + x_cor[j]) / 2
            # temp += d_l * f((j + 1.0 / 2) * d_l, y) * np.cos(0 * np.pi / b * ((j + 1.0 / 2) * d_l))
            temp += (x_cor[j + 1] - x_cor[j]) * f(x, y) * np.cos(0 * np.pi / b * x)
        fy[0][i] = temp / b  # 求出fy当p = 0时的值

    for k in range(1, num_eigen):
        fy[k] = np.zeros((num_seg_y,))
        for i in range(num_seg_y):
            temp = 0
            y = (y_cor[i] + y_cor[i + 1]) / 2

            for j in range(num_seg_x):
                x = (x_cor[j + 1] + x_cor[j]) / 2
                # temp += d_l * f((j + 1.0 / 2) * d_l, y) * np.cos(k * np.pi / b * ((j + 1.0 / 2) * d_l))
                temp += f(x, y) * (b / k / np.pi) * (
                            np.sin(k * np.pi / b * x_cor[j + 1]) - np.sin(k * np.pi / b * x_cor[j]))
            fy[k][i] = temp / b * 2  # 求出fy当p > 0时的值
    #print(fy)


    def fn(y, n):  # 定义fy函数
        for i in range(num_seg_y):
            if (y >= y_cor[i] and y <= y_cor[i + 1]):
                return fy[n][i]


    cn = np.zeros((num_eigen, num_eigen))

    for k in range(num_eigen):
        temp = 0
        for j in range(num_seg_y):
            y = (y_cor[j] + y_cor[j + 1]) / 2
            # temp += d_l * fn((j + 1.0 / 2) * d_l, k) * np.cos(0 * np.pi / a * ((j + 1.0 / 2) * d_l))
            temp += (y_cor[j + 1] - y_cor[j]) * fn(y, k)  # 对y方向进行积分

        cn[0, k] = -temp / a / (
                    np.power(0 * np.pi / a, 2) + np.power(k * np.pi / b, 2) + np.power(m, 2))  # p = 0时长表达式里的内容

    for p in range(1, num_eigen):  # p > 0时的cn
        for k in range(num_eigen):
            temp = 0
            for j in range(num_seg_y):
                y = (y_cor[j] + y_cor[j + 1]) / 2
                # temp += d_l * fn((j + 1.0 / 2) * d_l, k) * np.cos(p * np.pi / a * ((j + 1.0 / 2) * d_l))
                temp += fn(y, k) * (a / p / np.pi) * (
                            np.sin(p * np.pi / a * y_cor[j + 1]) - np.sin(p * np.pi / a * y_cor[j]))
            #
            cn[p, k] = -temp / a * 2 / (
                        np.power(p * np.pi / a, 2) + np.power(k * np.pi / b, 2) + np.power(m, 2))  # Why多乘以2
    #
    #end = time.time()
    #print(end - start)
    data = genfromtxt('temp0_1.txt', delimiter=',')  # 读入数据，生成一个二维列表
    x = data[:, 0:1]
    y = data[:, 1:2]
    temp = np.zeros((x.shape[0], 1))

    for p in range(num_eigen):
        for n in range(num_eigen):
            temp = temp + cn[p, n] * np.cos(n * np.pi / b * x) * np.cos(p * np.pi / a * y)
    max_Temp.append(float(max(temp)))
    print(max_Temp)
np.savetxt('max_T2.0.txt', max_Temp, delimiter=',')
end = time.time()
print(end - start)