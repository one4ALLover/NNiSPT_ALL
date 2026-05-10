# -*- coding: utf-8 -*-

"""
Created on Mon Feb 22 21:30:32 2025

@author: AM4
"""

# Попробуем обучить один нейрон на задачу классификации двух классов

import pandas as pd # библиотека pandas нужна для работы с данными
import matplotlib.pyplot as plt # matplotlib для построения графиков
import numpy as np # numpy для работы с векторами и матрицами


# Считываем данные 
# df = pd.read_csv('https://archive.ics.uci.edu/ml/'
#     'machine-learning-databases/iris/iris.data', header=None)

df = pd.read_csv('data.csv')


# смотрим что в них
print(df.head())

# три столбца - это признаки, четвертый - целевая переменная (то, что мы хотим предсказывать)

# выделим целевую переменную в отдельную переменную
y = df.iloc[:, 4].values

# так как ответы у нас строки - нужно перейти к численным значениям
y = np.where(y == "Iris-setosa", 1, -1)

# возьмем два признака, чтобы было удобне визуализировать задачу
X = df.iloc[:, [0,2,3]].values

# Признаки в X, ответы в y - постмотрим на плоскости как выглядит задача
plt.figure
plt.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2], color='red', marker='o')
plt.scatter(X[y==-1, 0], X[y==-1, 1], X[y==-1, 2], color='blue', marker='x')

# переходим к созданию нейрона
# функция нейрона:
# значение = w1*признак1+w2*признак2+w0
# ответ = 1, если значение > 0
# ответ = -1, если значение < 0

def neuron(w,x):
    if((w[0]+w[1]*x[0]+w[2]*x[1]+w[3]*x[2])>=0):
        predict = 1
    else: 
        predict = -1
    return predict

# проверим как это работает (веса зададим пока произвольно)
w = np.array([0, 0.1, 0.4, 0.6])
print(neuron(w,X[1])) # вывод ответа нейрона для примера с номером 1


# теперь создадим процедуру обучения
# корректировка веса производится по выражению:
# w_new = w_old + eta*x*y

# зададим начальные значения весов
w = np.random.random(4)
eta = 0.01  # скорость обучения
w_iter = [] # пустой список, в него будем добавлять веса, чтобы потом построить график
for xi, target, j in zip(X, y, range(X.shape[0])):
    predict = neuron(w,xi)   
    w[1:] += (eta * (target - predict)) * xi # target - predict - это и есть ошибка
    w[0] += eta * (target - predict)
    # каждую 10ю итерацию будем сохранять набор весов в специальном списке
    if(j%10==0):
        w_iter.append(w.tolist())

# посчитаем ошибки
sum_err = 0
for xi, target in zip(X, y):
    predict = neuron(w,xi) 
    sum_err += (target - predict)/2

print("Всего ошибок: ", sum_err)


# попробуем визуализировать процесс обучения
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# точки двух классов
ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2],
           color='red', marker='o', label='Setosa')
ax.scatter(X[y == -1, 0], X[y == -1, 1], X[y == -1, 2],
           color='blue', marker='x', label='Остальные')

# разделяющая плоскость: w0 + w1*x1 + w2*x2 + w3*x3 = 0
# x3 = -(w0 + w1*x1 + w2*x2) / w3
w = w_iter[-1]
x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
X1, X2 = np.meshgrid(x1, x2)
X3 = -(w[0] + w[1] * X1 + w[2] * X2) / w[3]

ax.plot_surface(X1, X2, X3, alpha=0.3, color='green')

ax.set_xlabel('Признак 1')
ax.set_ylabel('Признак 2')
ax.set_zlabel('Признак 3')
ax.legend()
plt.title('Разделяющая плоскость')
plt.show()


