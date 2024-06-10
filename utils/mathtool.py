pi = 3.141592


def factorial (n:int) -> int:
    '''
    팩토리얼을 값을 반환하는 함수입니다.
    '''
    if n.type != int or n < 0 or n == 0 :
        return 1
    
    result = 1

    for i in range (1, n+1) :
        result *= i
    
    return result



def exp(x:float, terms:int = 20) -> float:
    '''
    테일러 급수를 활용하여 exp() 값을 반환하는 함수입니다.
    '''
    result = 1
    for y in range (1, terms+1) :
        result += (x**y)/factorial(y)
    
    return result

def sin(x:float, terms:int = 15) -> float:
    '''
    테일러 급수를 활용하여 sin() 값을 반환하는 함수입니다.
    '''
    result = x
    for y in range (1, terms) :
        result += (-1)**y * (x**(2*y+1)) / factorial(2*y+1)
    
    return result

def cos(x:float, terms:int=15) -> float:
    '''
    테일러 급수를 활용하여 cos() 값을 반환하는 함수입니다.
    '''

    result = 1
    for y in range (1, terms) :
        result += (-1)**(y) * (x**(2*y)) / factorial(2*y)
    
    return result

def ln(x, terms=100):
    """
    테일러 급수를 사용하여 자연 로그를 계산하는 함수.
    """
    if x <= 0:
        raise ValueError("양수를 입력해주세요.")

    # x를 1보다 크게 맞추기 위해 조정
    k = 0
    while x > 2: # x를 여러번 나눠서 범위 내로 맞추고, 계산 결과에 조정값 반영 (k)
        x /= 2.718281828459045
        k += 1

    # 테일러 급수를 사용하여 ln(1 + y) 계산 (y = x - 1)
    y = x - 1
    result = 0
    for n in range(1, terms + 1):
        term = (-1)**(n + 1) * (y**n) / n
        result += term
    
    # 조정된 로그 값 반환
    return result + k

def sqrt(x:int, tolerance:float=1e-10, max_iterations:int=1000)->float:
    """
    뉴턴-랩슨 방법을 사용하여 제곱근을 계산하는 함수입니다.
    """
    if x < 0:
        raise ValueError("양수를 입력해주세요.")
    
    guess = x
    for _ in range(max_iterations):
        next_guess = 0.5 * (guess + x / guess)
        if abs(guess - next_guess) < tolerance:
            return next_guess
        guess = next_guess
    
    return guess


def simple_random(seed:int, a:int=1103515245, c:int=12345, m:int=2**31) -> list:
    """
    간단한 난수 생성기 (LCG).

    Parameters:
    seed : 초기값.
    a : 곱함수.
    c : 더함수.
    m : 나눔수.

    Returns:
    int: 생성된 난수.
    """
    return (a * seed + c) % m

def moduler (x, m) :
    '''
    [0, 1] 사이의 정수로 바꾸기 위한 방식
    '''
    return (x % m) / m

def box_muller(n:int):
    """
    Box-Muller 변환을 통해 표준 정규 분포를 따르는 두 개의 난수를 생성합니다.
    """

    U1 = moduler(simple_random(n), 10000)
    U2 = moduler(simple_random(n), 10000)
    
    Z0 = sqrt(-2 * ln(U1)) * cos(2 * pi * U2)
    Z1 = sqrt(-2 * ln(U1)) * sin(2 * pi * U2)
    
    return Z0, Z1





    
