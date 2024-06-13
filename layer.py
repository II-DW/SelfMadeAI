from utils.mathtool import box_muller, sqrt, exp, log
from utils.listutils import *

class Conv2d  :
    def __init__ (self, feature, input_size:tuple, filter_size:tuple, stride) :
        self.input_size = input_size
        self.feature = feature
        self.kernal_size = (filter_size[2], filter_size[3], filter_size[1])
        self.kernal = self.he_initialization_conv(filter_size)
        self.stride = stride
        
        # self.bias = []
        # for _ in range (self.input_size[0]) : 
        #     l = []
        #     for _ in range (self.input_size[1]) :
        #         l.append(zeros(self.input_size[2]-self.kernal_size[2]+1, self.input_size[2]-self.kernal_size[2]+1))
            
    
    def he_initialization_conv(self, filter_shape):
        """
        Box-Muller 변환을 사용하여 He 초기화를 수행합니다.

        filter_shape (tuple): 필터의 형태 (출력 채널 수, 입력 채널 수, 필터 높이, 필터 너비).

        """
        fan_in = filter_shape[0]* filter_shape[2] *filter_shape[3]  # 입력 채널 수 * 필터 높이 * 필터 너비
        std_dev = sqrt(2.0 / fan_in)
        
        filters  = []
        # 필터 초기화를 위한 배열 생성
        for _ in range (filter_shape[0]):
            l = []
            for _ in range (filter_shape[1]):
                l.append(zeros(filter_shape[2], filter_shape[3]))
            filters.append(l)

        # Box-Muller 변환을 통해 표준 정규 분포 난수를 생성하고 He 초기화 수행
        num_elements = fan_in * filter_shape[0]
        k = 0
        for m in range (filter_shape[0]) :
            for n in range (filter_shape[1]) :
                for i in range (filter_shape[2]) :
                    for j in range(filter_shape[3]) :
                        Z0, Z1 = box_muller(m*n*i*j)
                        if i < num_elements:    
                            filters[m][n][i][j] = Z0 * std_dev
                            k+=1
                        if i < num_elements:
                            filters[m][n][i][j] = Z1 * std_dev
                            k+=1

        
        return filters
    
    def update_kernal(self, amountL:list) -> list :
        self.kernal = subtractList(self.kernal, amountL) 
    
    def resize_output (self, X, x, y) :
        result = []
        idx = 0
        for _ in range (y) :
            l = []
            for _ in range (x) :
                try :
                    l.append(X[idx])
                except IndexError :
                    print(idx)
                    quit()
                idx+=1
            result.append(l)
        return result
    
    def forward(self, X) : #(1, 1, 28, 28), (16, 1, 3, 3) # (16, 1, 13, 13), (32, 16, 3, 3) # channel은 언제나 1일것이라고 가정함
        result = []
        for _ in range (self.feature) :
            for m in range (self.input_size[1]) :
                l = []
                for _ in range (self.input_size[1]) :
                    l.append(self.resize_output(im2col(X[m][0], self.kernal[m][0], self.input_size[2], self.input_size[3], self.kernal_size[0], self.kernal_size[1])[0], self.input_size[2]-self.kernal_size[0]+1,self.input_size[2]-self.kernal_size[1]+1))
                result.append(l)
        return result
        
    def updateW (self, dY, X, lr) :
        dW = [[[[0 for _ in range(self.kernal_size[0])] for _ in range(self.kernal_size[1])] for _ in range(len(X))] for _ in range(len(dY))]
        
        # 각 출력 채널 k에 대해
        for k in range(len(dY)):
            # 각 입력 채널 c에 대해
            for c in range(len(X)):
                # dY와 X에 대해 연산을 수행하고 행렬곱을 수행
                for i in range(len(dY[0][0])):  # 출력 높이
                    for j in range(len(dY[0][0][0])):  # 출력 너비
                        for m in range(self.kernal_size[0]):
                            for n in range(self.kernal_size[1]):
                                h = i * self.stride + m
                                w = j * self.stride + n 
                                if 0 <= h < len(X[c][0]) and 0 <= w < len(X[c][0][0]):
                                    dW[k][c][m][n] += dY[k][0][i][j] * X[c][0][h][w]
        
        dWdL = dotproduct4d(lr, dW)
        
        for i in range(len(dY)) :
            for j in range(len(X)) :
                self.kernal[i][j] = subtractList(self.kernal[i][j], dWdL[i][j])

    def updateX(self, dY) :
        
        N, C, H, W_in = self.input_size
        dX = [[[[0 for _ in range(W_in)] for _ in range(H)] for _ in range(C)] for _ in range(N)]

        # 각 입력 채널 c에 대해
        for c in range(C):
            # 각 배치 n에 대해
            for n in range(N):
                # 각 입력 높이 h에 대해
                for h in range(H):
                    # 각 입력 너비 w에 대해
                    for w in range(W_in):
                        # 각 출력 채널 k에 대해
                        for k in range(len(dY)):
                            # 각 출력 높이 i와 출력 너비 j에 대해
                            for i in range(len(dY[0][0])):
                                for j in range(len(dY[0][0][0])):
                                    h_in = h + i * self.stride 
                                    w_in = w + j * self.stride 
                                    if 0 <= h_in < self.kernal_size[0] and 0 <= w_in < self.kernal_size[1]:
                                        dX[n][c][h][w] += dY[k][0][i][j] * self.kernal[k][c][h_in][w_in]
        
        return dX
        
    def backward (self, gradient, X, lr) :
        self.updateW(gradient, X, lr)
        return self.updateX(gradient)

    
    
class MaxPool2d :
    def __init__(self, input_shape, pooling_size, stride) :
        self.input_shape  = input_shape
        self.pooling_size = pooling_size
        self.stride = stride

    def forward(self, X) :
        result = []
        result_idx = []

        for m in range(self.input_shape[0]) :
            L_0 = []
            L_0_idx = []
            
            for n in range(self.input_shape[1]) :
                L_1 = []
                L_1_idx = []
                for i in range(0,self.input_shape[2]-self.pooling_size[0],self.stride) :
                    L_2=[]
                    L_2_idx = []
                    for j in range(0, self.input_shape[3]-self.pooling_size[1], self.stride) :

                        max_num = X[m][n][i][j]
                        max_idx = (m, n, i, j)
                        for k1 in range (self.pooling_size[0]) :
                            for k2 in range (self.pooling_size[1]) :
                                if max_num < X[m][n][i+k1][j+k2] :
                                    max_idx = (m, n, i+k1, j+k2)

                        L_2.append(max_num)
                        L_2_idx.append(max_idx)

                    L_1.append(L_2)
                    L_1_idx.append(L_2_idx)
                L_0.append(L_1)
                L_0_idx.append(L_1_idx)
            result.append(L_0)
            result_idx.append(L_0_idx)
        self.max_idx_list = result_idx
        self.size = (len(result), len(result[0]), len(result[0][0]), len(result[0][0][0]))
        return result

    def backward (self, gradient) :
        
        result = []
        for _ in range(self.input_shape[0]) :
            l = []
            for j in range(self.input_shape[1]) :
                l.append(zeros(self.input_shape[2], self.input_shape[3]))
            result.append(l)

        m = 0
        i = 0
        j = 0
        k = 0
        for i in range(len(self.max_idx_list)) :
            for j in range(len(self.max_idx_list[0])) :
                for k in range(len(self.max_idx_list[0][0])) :
                    for m in range(len(self.max_idx_list[0][0][0])) :
                        idx = self.max_idx_list[i][j][k][m]
                        result[idx[0]][idx[1]][idx[2]][idx[3]] = gradient[i][j][k][m]
        
        return result

            

class FCN2d :
    def __init__(self, input_shape:tuple, output_shape:int) :
        self.input_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
        self.input_shape = input_shape
        self.output_size = output_shape
        self.shape = (self.input_size, output_shape)
        self.weights = self.he_initialization()
        self.bias = zeros(self.output_size, 1)
            
    def he_initialization(self):
        """
        Box-Muller 변환을 사용하여 He 초기화를 수행합니다.
        """
        n_in=1
        for n in self.input_shape :
            n_in *= n
        
        n_out = self.output_size

        std_dev = sqrt(2.0 / n_in)
        
        weights = zeros(n_in, n_out)
        num_elements = n_in * n_out

        i = 0
        for m in range(n_out) :
            for n in range (n_in) :
                Z0, Z1 = box_muller(m*n)
                if i < num_elements:
                    weights[m][n] = Z0 * std_dev
                    i += 1
                if i < num_elements:
                    weights[m][n] = Z1 * std_dev
                    i += 1
        return weights
    
    def flatten(self, X) :
        result = []
        for m in range(self.input_shape[0]) :
            for n in range(self.input_shape[1]) :
                for i in range(self.input_shape[2]) :
                    for j in range(self.input_shape[3]) :
                        result.append(X[m][n][i][j])
        return [result]

    def forward(self, X) :
        flattened_X = self.flatten(X)
        return AddList(Matrix_Multiplication(flattened_X, self.weights), self.bias)
    

    
    def updateW(self, gradient, X, lr) :
        dWdL = Matrix_Multiplication(transpose2d(X), transpose2d(gradient))
        dWdL = dotproduct(lr, dWdL)
        
        dWdL = transpose2d(dWdL)
        
        self.weights = subtractList(self.weights, dWdL)

    def updateB(self, gradient, lr) :
        self.bias = subtractList(self.bias, dotproduct(lr, gradient))
    
    def updateX(self, gradient, lr) :
        dXdL = Matrix_Multiplication(gradient, transpose2d(self.weights))
        dXdL = dotproduct(lr, dXdL)
        return dXdL    
    
    def resize(self, X) :
        result = []
        idx = 0
        for i in range(self.input_shape[0]) :
            L1 = []
            for j in range(self.input_shape[1]) :
                L2 =[]
                for k in range(self.input_shape[2]) :
                    L3 = []
                    for m in range(self.input_shape[3]) :
                        L3.append(X[0][idx])
                        idx +=1
                    L2.append(L3)
                L1.append(L2)
            result.append(L1)
        return result
        


    def backward(self, gradient, X, lr) :
        flattened_X = self.flatten(X)
        self.updateW(gradient, flattened_X, lr)
        self.updateB(gradient, lr)
        return self.resize(self.updateX(gradient, lr))
        
    
class FCN1d:
    def __init__(self, input_shape:int, output_shape:int) :
        self.input_size = input_shape
        self.output_size = output_shape
        self.shape = (self.input_size, output_shape)
        self.weights = self.he_initialization()
        self.bias = zeros(self.output_size, 1)

    def he_initialization(self):
        """
        Box-Muller 변환을 사용하여 He 초기화를 수행합니다.
        """
        n_in= self.input_size

        n_out = self.output_size

        std_dev = sqrt(2.0 / n_in)
        weights = zeros(n_in, n_out)
        
        num_elements = n_in * n_out
        
        i = 0
        for m in range (n_out) :
            for n in range (n_in) :
                Z0, Z1 = box_muller(m*n)
                if i < num_elements:
                    weights[m][n] = Z0 * std_dev
                    i+=1
                if i < num_elements:
                    weights[m][n] = Z1 * std_dev
                    i+=1
        
        return weights

    def forward(self, X) :
        return AddList(Matrix_Multiplication(X, self.weights), self.bias)
    
    def updateW (self, gradient, X, lr) :
        dWdL = Matrix_Multiplication(transpose2d(X), transpose2d(gradient)) 
        dWdL = dotproduct(lr, dWdL)
        self.weights = subtractList(self.weights, transpose2d(dWdL))

    def updateB (self, gradient, lr) :
        self.bias = subtractList(self.bias, dotproduct(lr, gradient))

    def updateX (self, gradient) :
        return Matrix_Multiplication(gradient, transpose2d(self.weights))

    
    def backward(self, gradient, X, lr) :
        self.updateW(gradient, X, lr)
        self.updateB(gradient, lr)
        return self.updateX(gradient)

class softmax :
    def __init__ (self, input_len) :
        self. input_len = input_len

    def forward (self, X) :
        result = zeros(1, self.input_len)
        exp_l = [exp(X[0][i]) for i in range(self.input_len)]
        exp_sum = sum(exp_l)
        for n in range(self.input_len) :
            result[n] = exp_l[n] / exp_sum
        return result
    
class CrossEntropyLoss :
    def __init__(self, num_classes:tuple) :
        self.num_classes = num_classes
    
    def forward(self, X, Y) :
        result = 0
        for i in range (self.num_classes) :
            result -= Y[i] * log(X[i], 10)
        return result
    
    def backward(self, X, Y) :
        return subtractList(X, Y)
