# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(67)
# Для работы в Windows с multiprocessing нужно обернуть код в if __name__ == '__main__'
if __name__ == '__main__':

    # Устанавливаем разрешение на использование дополнительных библиотек
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Сначала определим на каком устройстве будем работать - GPU или CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Подготовка трансформаций данных
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Загружаем обучающий и тестовый наборы данных
    # Убедитесь, что структура папок для каждого класса правильная: 
    # "train/кошки", "train/хомяки", "train/капибары" и так же для тестовых данных.
    train_dataset = torchvision.datasets.ImageFolder(root='D:/UNIK/6 семестр/Нейронные сети для решения практических задач/5LAB_DATA/data/train', transform=data_transforms)
    test_dataset = torchvision.datasets.ImageFolder(root='D:/UNIK/6 семестр/Нейронные сети для решения практических задач/5LAB_DATA/data/test', transform=data_transforms)

    # Загружаем данные через DataLoader
    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Используем AlexNet, предобученную на ImageNet
    net = torchvision.models.vgg19(pretrained=True)

    # Замораживаем веса предобученной части сети (не обучаем их)
    for param in net.parameters():
        param.requires_grad = False

    # Меняем классификатор на три класса
    num_classes = 3  # Обновляем количество классов
    new_classifier = net.classifier[:-1]  # убираем последний слой
    new_classifier.add_module('fc', nn.Linear(4096, num_classes))  # добавляем новый слой с тремя классами
    net.classifier = new_classifier  # заменяем классификатор в сети

    # Переводим модель на устройство
    net = net.to(device)

    # Задаем функцию потерь и оптимизатор
    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    # Обучение модели
    num_epochs = 5
    save_loss = []

    start_time = time.time()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Прямой проход
            outputs = net(images)

            # Вычисление потерь
            loss = lossFn(outputs, labels)

            # Обратный проход (вычисление градиентов)
            optimizer.zero_grad()
            loss.backward()

            # Шаг оптимизации
            optimizer.step()

            save_loss.append(loss.item())

            # Вывод диагностической информации
            if i % 100 == 0:
                print(f'Эпоха {epoch + 1}/{num_epochs}, Шаг {i}, Ошибка: {loss.item()}')

    print("Время обучения:", time.time() - start_time)

    # Площадка для отображения графика потерь
    plt.figure()
    plt.plot(save_loss)
    plt.title('Изменение потерь на обучении')

    # Тестирование модели
    correct_predictions = 0
    num_test_samples = len(test_dataset)

    with torch.no_grad():  # Отключаем вычисление градиентов, так как делаем только прямой проход
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = net(images)  # Прогоняем данные через модель
            _, pred_class = torch.max(pred.data, 1)  # Выбираем класс с максимальной вероятностью
            correct_predictions += (pred_class == labels).sum().item()

    print(f'Точность модели: {100 * correct_predictions / num_test_samples:.2f}%')

    # Сохранение модели
    torch.save(net.state_dict(), 'VGG19.ckpt')

    # Прогнозирование и отображение картинок с предсказаниями
    inputs, classes = next(iter(test_loader))
    pred = net(inputs.to(device))
    _, pred_class = torch.max(pred.data, 1)

    # Визуализация изображений и их предсказанных классов
    for i, j in zip(inputs, pred_class):
        img = i.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(train_dataset.classes[j])
        plt.pause(2)  # Ожидание 2 секунды между изображениями