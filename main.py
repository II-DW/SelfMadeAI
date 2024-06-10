from data.data_loader import dataPreprocess
from layer import Conv2d, MaxPool2d, FCN
D = dataPreprocess("./data/mnist_train.txt", "./data/mnist_test.txt")
train_label, train_data, test_label, test_data = D.get_item()


Layer = ['Conv2d', 'MaxPooling', 'Conv2d', 'MaxPooling', 'FCN']

epochs = 1000
learning_rate = 0.0001

Layer_List = []
Layer_List.append(Conv2d(32, (1, 1, 28, 28), 3, 1))
Layer_List.append(MaxPool2d((1, 32, 14, 14), (2, 2),2))
Layer_List.append(Conv2d(64, (1, 32, 14, 14), 3, 1))
Layer_List.append(MaxPool2d())
Layer_List.append(FCN())
Layer_List.append(FCN())
