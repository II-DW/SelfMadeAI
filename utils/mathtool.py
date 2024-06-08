class Math :

    def __init__ (self) :
        pass

    def Matrix_Multiplication(self, A:list, B:list) -> list:
        '''
        함수의 행렬 곱을 반환하는 함수입니다. \n
        2차원 행렬 곱만 지원합니다. \n
        입력 행렬이 행렬 곱을 할 수 있는 크기여야 합니다. ex) (a, b)와 (b, c) 크기의 행렬

        A : 2차원 리스트 (크기 (a, b)) 2, 4
        B : 2차원 리스트 (크기 (b, c)) 4, 9
        반환값 : 2차원 리스트 (크기 (a, c))
        '''

        result = []

        a = len(A)
        b = len(A[0])
        b1 = len(B)
        c = len(B[0])

        if b == b1 :
            pass
        else :
            print("입력 행렬의 크기가 행렬곱을 하기에 적당하지 않습니다.")
            return ["error"]
        
        for i in range (a) :
            L = []
            for j in range (c) :
                s_num = 0
                for k in range (b):
                    s_num += A[i][k] * B[k][j]
                L.append(s_num)
            result.append(L)

        return result
                





    
