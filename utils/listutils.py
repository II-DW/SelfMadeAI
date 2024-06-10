def Matrix_Multiplication(A:list, B:list) -> list:
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


def subtractList (A:list, B:list) -> list :
    '''
    두 2차원 행렬을 입력받았을 때, 뺄셈을 계산하는 함수
    크기는 같아야함.
    '''
    result = []
    try :
        for i in range (len(A)) :
            l = []
            for j in range (len(B)) :
                l.append(A[i][j] - B[i][j])
            result.append(l)
        return result
    except Exception as e :
        print("Error :", e)
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
            for j in range (len(B)) :
                l.append(A[i][j] + B[i][j])
            result.append(l)
        return result
    except Exception as e :
        print("Error :", e)
        return result

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
  for n in range ((H - k1 + 1)) :
    for m in range ((W - k2 + 1)) :
      L = [X[n][m], X[n][m+1], X[n+1][m], X[n+1][m+1]]
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
  return result


def im2col (X, w, H, W, k1, k2) :
  '''
  im2col 연산 함수
  '''
  Xc = return_Xc(X, H, W, k1, k2)
  Wc = return_Wc(w, k1, k2)
  producted_array = Matrix_Multiplication(Xc, Wc)
  return producted_array

            