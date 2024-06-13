from data.data_loader import dataPreprocess
from layer import Conv2d, MaxPool2d, FCN2d, FCN1d, softmax, CrossEntropyLoss

D = dataPreprocess("./data/mnist_train.txt", "./data/mnist_test.txt")
train_label, train_data, test_label, test_data = D.get_item()
print('Step 1.Done Data Load...')

epochs = 100
learning_rate = 0.0001

Layer_List = []
Layer_List.append(Conv2d(16, (1, 1, 28, 28), (16, 1, 3, 3), 1))
print("Done Layer...", 1)
Layer_List.append(MaxPool2d((16, 1, 26, 26), (2, 2),2))
print("Done Layer...", 2)
Layer_List.append(Conv2d(32, (16, 1, 12, 12), (32, 16, 3, 3), 1))
print("Done Layer...", 3)
Layer_List.append(MaxPool2d((32, 1,  10, 10), (2, 2),2))
print("Done Layer...", 4)
Layer_List.append(FCN2d((32, 1, 4, 4), 32))    
print("Done Layer...", 5)
Layer_List.append(FCN1d(32, 10))    
print("Done Layer...", 6)
Layer_List.append(softmax(10))    
print("Done Layer...", 7)
print("Step 2. Done Layer Setting")

Loss = CrossEntropyLoss(10)

# write_data.py



for epoch in range(epochs) :
    for i in range (len(train_data)) :
        label, img = train_label[i], train_data[i] 

        X_list = []
        # 순전파
        x = Layer_List[0].forward([[img]])
        X_list.append(x)
        x = Layer_List[1].forward(x)
        X_list.append(x)
        x = Layer_List[2].forward(x)    
        X_list.append(x)
        x = Layer_List[3].forward(x)
        X_list.append(x)
        x = Layer_List[4].forward(x)
        X_list.append(x)
        x = Layer_List[5].forward(x)
        X_list.append(x)
        x = Layer_List[6].forward(x)
        X_list.append(x)

        # 손실 계산
        Y = [0 for _ in range(10)]
        Y[label-1] = 1 # one-hot encoding
        loss = Loss.forward(Y, x)
        print(loss)

        # 역전파
        gradient = Loss.backward([x], [Y]) # Cross Entrophy Loss + softmax 역전파, 출력은 (1, 10) 크기의 벡터
        dLdX = Layer_List[5].backward(gradient, X_list[4], learning_rate) 
        dLdX = Layer_List[4].backward(dLdX, X_list[3], learning_rate) 
        dLdX = Layer_List[3].backward(dLdX) 
        dLdX = Layer_List[2].backward(dLdX, X_list[1], learning_rate) 
        dLdX = Layer_List[1].backward(dLdX) 
        dLdX = Layer_List[2].backward(dLdX, [[img]], learning_rate) 

        f = open('./data.txt', 'a')
        f.write(str(loss) + '\n')
        f.close()
    f = open('./data.txt', 'a')
    f.write(epoch+"learning" + '\n\n')
    f.close()
    print("Step 3-"+str(epoch)+". Learning...")

