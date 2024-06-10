from data.data_loader import dataPreprocess
from layer import Conv2d, MaxPool2d, FCN2d, FCN1d

D = dataPreprocess("./data/mnist_train.txt", "./data/mnist_test.txt")
train_label, train_data, test_label, test_data = D.get_item()
print('Step 1.Done Data Load...')

epochs = 1000
learning_rate = 0.0001

Layer_List = []
Layer_List.append(Conv2d(32, (1, 1, 28, 28), (32, 1, 3, 3), 1))
print("Done Layer...", 1)
Layer_List.append(MaxPool2d((1, 32, 14, 14), (2, 2),2))
print("Done Layer...", 2)
Layer_List.append(Conv2d(64, (1, 32, 14, 14), (64, 32, 3, 3), 1))
print("Done Layer...", 3)
Layer_List.append(MaxPool2d((1, 64, 14, 14), (2, 2),2))
print("Done Layer...", 4)
Layer_List.append(FCN2d((1, 64, 7, 7), 32))    
print("Done Layer...", 5)
Layer_List.append(FCN1d(128,10))
print("Done Layer...", 6)

print("Step 2. Done Layer Setting")

for epoch in range(epochs) :
    for i in range (len(train_data)) :
        label, img = train_label[i], train_data[i] 

        x = Layer_List[0].forward([[img]])
        print(x)
        x = Layer_List[1].forward(x)
        print(x)
        x = Layer_List[2].forward(x)
        print(x)
        x = Layer_List[3].forward(x)
        print(x)
        x = Layer_List[4].forward(x)
        print(x)
        x = Layer_List[5].forward(x)
        print(x)
    print("Step 3-"+str(epoch)+". Learning...")