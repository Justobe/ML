# -*- coding: utf-8 -*-
# @Author  :yenming
# @Time    :2018/4/11  16:10

import numpy as np
import math
from scipy.optimize import fmin,fminbound

def obj_function(x):
    return x + 10 * math.sin(5*x) -7 * math.cos(4*x)

#二进制转十进制
def bin2dec(bin):
    res = 0
    max_index = len(bin) - 1
    for i in range(len(bin)):
        if bin[i] == 1:
            res = res + 2**(max_index-i)
    return res

#获取特定精度下，染色体长度
def get_encode_length(low_bound,up_bound,precision): # e.g 0,9,0.01
    divide = (up_bound - low_bound)/precision
    for i in range(10000):
        if 2**i < divide < 2**(i + 1):
            return i+1
    return -1

#将二进制的染色体解码成[low,up]空间内的数
def decode_chromosome(low_bound,up_bound, length, chromosome):
    return low_bound + bin2dec(chromosome)*(up_bound - low_bound)/(2**length -1)

#定义初始染色体
def intial_population(length,population_size):
    chromosomes = np.zeros((population_size,length),dtype=np.int8)
    for i in range(population_size):
        chromosomes[i] = np.random.randint(0,2,length)              #随机数[0,2)之间的整数
    return chromosomes


##########hyperparameter##############
# f(x) = x + 10*sin(5x) -7*cos(4x)
population_size = 500   #种群大小
iselitist = True        #是否精英选择
generations = 1000      #演化多少代
low_bound = 0           #区间下界
up_bound = 9            #区间上界
precision = 0.0000001      #精度
chromosome_length = get_encode_length(low_bound,up_bound,precision)  #染色体长度
populations = intial_population(chromosome_length,population_size)   #初始种群
best_fit = 0            #获取到的最大的值
best_generation = 0     #获取到最大值的代数
best_chromosome = [0 for x in range(population_size)]               #获取到最大值的染色体
fitness_average = [0 for x in range(generations)]                   #种群平均适应度
cross_rate = 0.6                                                    #基因交叉率
mutate_rate = 0.01                                                  #基因变异率
######################################

#计算种群中每个染色体的适应度函数值(在这个问题中，适应度函数就是目标函数)
def fitness(populations):
    fitness_val = [0 for x in range(population_size)]
    for i in range(population_size):
        fitness_val[i] = obj_function(decode_chromosome(low_bound,up_bound,chromosome_length,populations[i]))
    return fitness_val

#对种群染色体根据适应度函数进行排序，适应度函数值高的在最后
def rank(fitness_val,populations,cur_generation):
    global best_fit,best_generation,best_chromosome
    global fitness_average
    fitness_sum = [0 for x in range(len(populations))]              #初始化种群累计适应度
    #population_size,length = populations.shape
    for i in range(len(populations)):                               #冒泡排序按照种群适应度从小到大
        min_index = i
        for j in range(i+1,population_size):
            if fitness_val[j] < fitness_val[min_index]:
                min_index = j
        if min_index!=i:
            tmp = fitness_val[i]
            fitness_val[i] = fitness_val[min_index]
            fitness_val[min_index] = tmp

            tmp_list = np.zeros(chromosome_length)
            for k in range(chromosome_length):
                tmp_list[k] = populations[i][k]
                populations[i][k] = populations[min_index][k]
                populations[min_index][k] = tmp_list[k]

    #########种群适应度从小到大排序完毕#########
    for l in range(len(populations)):                               #获取种群累计适应度
        if l == 1:
            fitness_sum[l] = fitness_val[l]
        else:
            fitness_sum[l] = fitness_val[l] + fitness_val[l-1]

    fitness_average[cur_generation] = fitness_sum[-1]/population_size   #每一代的平均适应度，在这个算法程序中没有使用到，仅作记录

    if fitness_val[-1] > best_fit:                                  #更新最佳适应度及其对应的染色体
        best_fit = fitness_val[-1]
        best_generation = cur_generation
        for m in range(chromosome_length):
            best_chromosome[m] = populations[-1][m]
    return fitness_sum

#根据当前种群，按照轮盘法选择新一代染色体
def select(populations,fitness_sum,iselitist): #轮盘选择法，实现过程可看为二分查找
    population_new = np.zeros((population_size, chromosome_length), dtype=np.int8)
    for i in range(population_size):
        rnd = np.random.rand()*fitness_sum[-1]
        first = 0
        last = population_size-1
        mid = (first+last)//2
        idx = -1
        while first <= last:
            if rnd >fitness_sum[mid]:
                first = mid
            elif rnd < fitness_sum[mid]:
                last = mid
            else:
                idx = mid
                break
            if last - first == 1:
                idx = last
                break
            mid = (first + last) // 2

        for j in range(chromosome_length):
            population_new[i][j] = populations[idx][j]
    if iselitist == True: #是否精英选择，精英选择强制保留最后一个染色体(适应度函数值最高)
        p = population_size - 1
    else:
        p = population_size
    for k in range(p):
        for l in range(chromosome_length):
            populations[k][l] = populations[k][l]

#基因交叉
def crossover(populations):
    for i in range(0,population_size,2):
        rnd = np.random.rand()
        if rnd < cross_rate:
            rnd1 = int(math.floor(np.random.rand()*chromosome_length))
            rnd2 = int(math.floor(np.random.rand()*chromosome_length))
        else:
            continue
        if rnd1 <= rnd2:
           cross_position1 = rnd1   #这里我选择了一个基因片段，进行两点的交叉
           cross_position2 = rnd2
        else:
           cross_position1 = rnd2
           cross_position2 = rnd1
        for j in range(cross_position1,cross_position2):
            tmp = populations[i][j]
            populations[i][j] = populations[i+1][j]
            populations[i+1][j] = tmp
#基因变异
def mutation(populations):
    for i in range(population_size):
        rnd = np.random.rand()
        if rnd < mutate_rate:
            mutate_position = int(math.floor(np.random.rand()*chromosome_length))
        else:
            continue
        populations[i][mutate_position] = 1 - populations[i][mutate_position]

#演化generations代
for g in range(generations):
    print("generation {} ".format(g))
    fitness_val = fitness(populations)
    fitness_sum = rank(fitness_val,populations,g)
    select(populations,fitness_sum,iselitist)
    crossover(populations)
    mutation(populations)
    print("best chromosome", best_chromosome)
    print("best_generation", best_generation)
    print("best_fit", best_fit)

print("####################Done######################")
print("best chromosome",best_chromosome)
print("best_generation",best_generation)
print("best_fit",best_fit)



print("####################Actual###################")


def func(x):#使用fmincound 将函数取负，求最小值
    return -obj_function(x)


min_global=fminbound(func,0,9)#这个区域的最小值
print("The actual max value",-func(min_global))
x = [0,1,2,3,4,5,6,7,8,9]
y = [i + 10 * np.sin(i*5) -7 * np.cos(i*4) for i in x]
import matplotlib.pyplot as plt
plt.plot(x,y)
plt.show()
