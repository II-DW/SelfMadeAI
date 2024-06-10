from utils.mathtool import box_muller, sqrt
from utils.listutils import zeros, subtractList, im2col, Matrix_Multiplication, AddList

class Conv2d  :
    def __init__ (self, feature, input_size:tuple, filter_size:tuple, stride) :
        self.input_size = input_size
        self.feature = feature
        self.kernal_size = (filter_size[2], filter_size[3], filter_size[1])
        self.kernal = self.he_initialization_conv(filter_size)
        self.stride = stride
    
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
        
        return result

class FCN2d :
    def __init__(self, input_shape:tuple, output_shape:int) :
        self.input_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
        self.input_shape = input_shape
        self.output_size = output_shape
        self.shape = (self.input_size, output_shape)
        self.weights = self.he_initialization()
        self.bias = zeros(self.input_size, self.output_size)
            
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
    
class FCN1d:
    def __init__(self, input_shape:int, output_shape:int) :
        self.input_size = input_shape
        self.output_size = output_shape
        self.shape = (self.input_size, output_shape)
        self.weights = self.he_initialization()
        self.bias = zeros(self.input_size, self.output_size)

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