from data.data_loader import dataPreprocess

D = dataPreprocess("./data/mnist_train.txt", "./data/mnist_test.txt")
train_label, train_data, test_label, test_data = D.get_item()


Layer = ['Conv2d', 'MaxPooling', 'Conv2d', 'MaxPooling', 'FCN']





epochs = 1000
learning_rate = 0.0001
