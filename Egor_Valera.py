# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:48:33 2026

@author: egorr
"""

import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd

class NNet(nn.Module):
    # для инициализации сети на вход нужно подать размеры (количество нейронов) входного, скрытого и выходного слоев
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        # nn.Sequential - контейнер модулей
        # он последовательно объединяет слои и позволяет запускать их одновременно
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size),
                                    nn.Tanh(),# слой линейных сумматоров# функция активации
                                    nn.Linear(hidden_size, out_size),
                                    nn.Sigmoid()
                                    )
    # прямой проход    
    def forward(self,X):
        pred = self.layers(X)
        return pred
    
    
df = pd.read_csv('dataset_simple.csv')
X_np = df.iloc[:,0:2].values
X_np = (X_np - X_np.min(axis=0)) / (X_np.max(axis=0) - X_np.min(axis=0))
X = torch.tensor(X_np, dtype=torch.float32)

print(X)
y = torch.Tensor(df.iloc[:, 2].values).reshape(-1,1)

inputSize = X.shape[1] # количество признаков задачи 
hiddenSizes = 5 #  число нейронов скрытого слоя 
outputSize = 1

# Создаем экземпляр нашей сети
net = NNet(inputSize,hiddenSizes,outputSize)

# Веса нашей сети содержатся в net.parameters() 
for param in net.parameters():
    print(param)

# Можно вывести их с названиями
for name, param in net.named_parameters():
    print(name, param)


# откл град
with torch.no_grad():
    pred = net.forward(X)

# Для обучения нам понадобится выбрать функцию вычисления ошибки
lossFn = nn.BCELoss()

# и алгоритм оптимизации весов
# при создании оптимизатора в него передаем настраиваемые параметры сети (веса)
# и скорость обучения
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)

epohs = 1500
for i in range(0,epohs):
    pred = net.forward(X)   #  прямой проход - делаем предсказания
    loss = lossFn(pred, y)  #  считаем ошибу 
    optimizer.zero_grad()   #  обнуляем градиенты 
    loss.backward()
    optimizer.step()
    if i%10==0:
       print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

    
# Посчитаем ошибку после обучения
with torch.no_grad():
    pred = net.forward(X)

print(pred)

pred = torch.Tensor(np.where(pred >=0.5, 1, 0).reshape(-1,1))
err = sum(abs(y-pred))/2
print('\nОшибка (количество несовпавших ответов): ')
print(err)

print(pred)
