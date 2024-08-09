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
    # print(Power)
    # print(FUnit)
    num_power = Power.shape[1]
    #print(Power.shape[1])
    # print(FUnit.shape[0])

    y_cor = set()

    for i in range(num_power):
        y_cor.add(FUnit[i, 4])
    y_cor.add(a)

    y_cor = list(y_cor)
    y_cor.sort()
    # print(y_cor)
    def f(x, y):
        for i in range(num_power):
            if (x >= FUnit[i, 3] and x <= FUnit[i, 3] + FUnit[i, 1] and y >= FUnit[i, 4] and y <= FUnit[i, 4] + FUnit[
                i, 2]):
                return -Power[1, i] / (FUnit[i, 1] * FUnit[i, 2] * 0.00015) / K/K / gamma - m * m * T0

    num_eigen = 20

    num_l = 200

    fy = {}

    num_seg = len(y_cor) - 1
    fy[0] = np.zeros((num_seg,))
    for i in range(num_seg):
        temp = 0
        d_l = b / num_l
        y = (y_cor[i] + y_cor[i + 1]) / 2
        for j in range(num_l):
            temp += d_l * f((j + 1.0 / 2) * d_l, y) * np.cos(0 * np.pi / b * ((j + 1.0 / 2) * d_l))

        fy[0][i] = temp / b

    for k in range(1, num_eigen):
        fy[k] = np.zeros((num_seg,))
        for i in range(num_seg):
            temp = 0
            d_l = b / num_l
            y = (y_cor[i] + y_cor[i + 1]) / 2
            for j in range(num_l):
                temp += d_l * f((j + 1.0 / 2) * d_l, y) * np.cos(k * np.pi / b * ((j + 1.0 / 2) * d_l))

            fy[k][i] = temp / b * 2
            # print(fy)

    def fn(y, n):
        for i in range(num_seg):
            if (y >= y_cor[i] and y <= y_cor[i + 1]):
                return fy[n][i]


    cn = np.zeros((num_eigen, num_eigen))

    for k in range(num_eigen):
        temp = 0
        d_l = a / num_l
        for j in range(num_l):
            temp += d_l * fn((j + 1.0 / 2) * d_l, k) * np.cos(0 * np.pi / a * ((j + 1.0 / 2) * d_l))

        cn[0, k] = -temp / a / (np.power(0 * np.pi / a, 2) + np.power(k * np.pi / b, 2) + np.power(m, 2))

    for p in range(1, num_eigen):
        for k in range(num_eigen):
            temp = 0
            d_l = a / num_l
            for j in range(num_l):
                temp += d_l * fn((j + 1.0 / 2) * d_l, k) * np.cos(p * np.pi / a * ((j + 1.0 / 2) * d_l))

            cn[p, k] = -temp / a * 2 / (np.power(p * np.pi / a, 2) + np.power(k * np.pi / b, 2) + np.power(m, 2))
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
np.savetxt('max_T_old.txt', max_Temp, delimiter=',')
end = time.time()
print(end - start)