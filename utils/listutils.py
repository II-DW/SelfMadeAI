def Matrix_Multiplication(A: list, B: list) -> list:
    '''
    함수의 행렬 곱을 반환하는 함수입니다.
    2차원 행렬 곱만 지원합니다.
    입력 행렬이 행렬 곱을 할 수 있는 크기여야 합니다. ex) (a, b)와 (b, c) 크기의 행렬

    A : 2차원 리스트 (크기 (a, b))
    B : 2차원 리스트 (크기 (b, c))
    반환값 : 2차원 리스트 (크기 (a, c))
    '''
    
    a = len(A) # 1
    b = len(A[0]) # 9
    b = len(B[0]) # 9
    c = len(B) # 676

    # 결과 행렬 초기화
    result = zeros(c, a)

    for i in range(a):
        for j in range(c):
            s_num = 0
            for k in range(b):
                s_num += A[i][k] * B[j][k]
            result[i][j] = s_num

    return result



def subtractList (A:list, B:list) -> list :
    '''
    두 2차원 행렬을 입력받았을 때, 뺄셈을 계산하는 함수
    크기는 같아야함.
    '''
    result = []
    try :
        for i in range (len(A)) :
            l = []
            for j in range (len(A[0])) :
                l.append(A[i][j] - B[i][j])
            result.append(l)
        return result
    except Exception as e :
        print("Error :", e)
        raise IndexError 


def AddList (A:list, B:list) -> list :
    '''
    두 2차원 행렬을 입력받았을 때, 덧셈을 계산하는 함수
    크기는 같아야함.
    '''
    result = []
    try :
        for i in range (len(A)) :
            l = []
            for j in range (len(A[0])) :
                l.append(A[i][j] + B[i][j])
            result.append(l)
        return result
    except Exception as e :
        print("Error :", e)
        raise IndexError

def zeros (width:int, height:int) -> list :
    '''
    크기와 높이를 받았을 때, 0으로 해당 크기의 행렬을 만들어주는 함수입니다.
    (np.zeros())
    '''
    result = []
    for _ in range (height) :
        result.append([0 for _ in range (width)])
    return result


def makeListSum (L:list) -> list :
    '''
    2차원 행렬을 입력받았을 때, 모든 원소의 합이 1이 되도록 만들어주는 함수입니다.
    '''
    result = L
    
    s_num = 0
    for l in L :
        s_num += sum(l)
    
    for m in range(len(result)) :
        for n in range(len(result[0])) :
            result[m][n] /= s_num
            
    return result



def return_Xc(X:list, H:int, W:int, k1:int, k2:int) -> list:
  '''
  im2col 연산용 함수 1
  X를 입력했을 때, X_c를 반환하는 함수
  '''
  result = []

  for n in range (H - k1 + 1) :
      for m in range (W - k2 + 1) :
          L = []
          for l in X[m:m+k2] :
              for e in l[n:n+k1] :
                  L.append(e)
          result.append(L)
  
      
      
  return result
     

def return_Wc(W:list, k1:int, k2:int) :
  '''
  im2col 연산용 함수 2
  W를 입력했을 때, W_c를 반환하는 함수
  '''
  result = []
  for i in range(k2):
    for j in range(k1):
      result.append(W[i][j])
  return [result]

def im2col (X, w, H, W, k1, k2) :
  '''
  im2col 연산 함수
  '''
  Xc = return_Xc(X, H, W, k1, k2)
  Wc = return_Wc(w, k1, k2)
  producted_array = Matrix_Multiplication(Wc, Xc)
  return producted_array

def transpose2d(X:list) -> list:
    '''
    2차원 행렬의 전치행렬 반환
    '''
    W = len(X[0])
    H = len(X)
    result = zeros(H, W)
    for i in range(W) :
        for j in range (H) :
            result[i][j] = X[j][i]
    return result

def dotproduct(scalar:float, matrix:list) -> list :
    '''
    행렬 x 스칼라 곱
    '''
    result = matrix
    for i in range(len(matrix)) :
        for j in range(len(matrix[0])) :
            result[i][j] = scalar*matrix[i][j]
    return result

def dotproduct4d(scalar:float, matrix:list) -> list :
    '''
    행렬 x 스칼라 곱
    '''
    result = matrix
    for i in range(len(matrix)) :
        for j in range(len(matrix[0])) :
            for k in range(len(matrix[0][0])) :
                for m in range(len(matrix[0][0][0])) :
                    result[i][j][k][m] = scalar*matrix[i][j][k][m]
    return result