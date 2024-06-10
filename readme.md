# 모듈 사용하지 않고 CNN 구현하기

<div align="center">
<img src="https://img.shields.io/badge/Python-black?style=flat&logo=python&logoColor=#3776AB"/> 
</div>

## 목차
- [Model Information](#model-information)
  * [Layer 1](#layer-1)
  * [Layer 2](#layer-2)
  * [Layer 3](#layer-3)
  * [Layer 4](#layer-4)
  * [Layer 5](#layer-5)
- [File Structure](#File-Structure)
  * [data](#data)
  * [model](#model)
  * [test](#test)
  * [utils](#utils)
  * [others](#others)
- [Learning](#learning)
  * [순전파(Forward propagation)](#순전파forward-propagation)
  * [역전파(Backward propagation)](#역전파backward-propagation)
- [Reference & Data](#reference--data)



# Model Information

## Layer 1
~~~ JSON
{
  "Type" : "Convolutional Layer",
  "Input Shape" : [28, 28, 1],
  "Output Shape" : [28, 28, 32],
  "Number Of Filters" : 32,
  "Filter Size" : [3, 3],
  "Stride" : 1, 
  "Padding" : "same",
  "Activation Function" : "Leaky ReLU"
}
~~~
## Layer 2

~~~ JSON
{
  "Type" : "Pooling Layer",
  "Input Shape" : [28, 28, 32],
  "Output Shape" : [14, 14, 32],
  "Pooling Type" : "Max Pooling",
  "Pooling Size" : [2, 2],
  "Stride" : 2
}
~~~

## Layer 3

~~~ JSON
{
  "Type" : "Convolutional Layer",
  "Input Shape" : [14, 14, 32],
  "Output Shape" : [14, 14, 64],
  "Number Of Filters" : 64,
  "Filter Size" : [3, 3],
  "Stride" : 1, 
  "Padding" : "same",
  "Activation Function" : "Leaky ReLU"
}
~~~

## Layer 4

~~~ JSON
{
  "Type" : "Pooling Layer",
  "Input Shape" : [14, 14, 64],
  "Output Shape" : [7, 7, 64],
  "Pooling Type" : "Max Pooling",
  "Pooling Size" :[2, 2],
  "Stride" : 2
}
~~~

## Layer 5

~~~ JSON
{
  "Type" : "Fully Connected Layer",
  "Input Shape" : [7, 7, 64],
  "Output Shape" : [128],
  "Number Of Neurons" : 128 , 
  "Activation Function" : "Leaky ReLU"
}
~~~

## Layer 6

~~~ JSON
{
  "Type" : "Output Layer",
  "Input Shape" : [128],
  "Output Shape" : 10,
  "Number Of Neurons" : 10, 
  "Activation Function" : "softmax"
}
~~~

# File-Structure

## data
**MNIST 데이터 및 데이터 처리 클래스** <br>
**MNIST Data & Data Process Class**
## model
**신경망 층 구성 및 model 구성 클래스** <br>
**Neuron Layer & Model Composition Class**
## test
**test용 코드들** <br>
**Code For Test**
## utils
**수학적 계산 도구들** <br>
**Math Tools**
## others

# Learning

## 순전파(Forward propagation)
순전파 연산 과정

## 역전파(Backward propagation)
역전파 연산 과정

## Reference & Data
[MNIST in CSV - kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

## Used Math Principle


### 테일러 급수

$$
e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}
$$

$$
\sin(x) = \sum_{n=0}^{\infty} (-1)^n \frac{x^{2n+1}}{(2n+1)!}
$$

$$
\cos(x) = \sum_{n=0}^{\infty} (-1)^n \frac{x^{2n}}{(2n)!}
$$



$$
\ln(1 + x) = \sum_{n=1}^{\infty} (-1)^{n+1} \frac{x^n}{n}
$$


### 오차역전파



$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot B^T
$$

$$
\frac{\partial L}{\partial B} = A^T \cdot \frac{\partial L}{\partial C}
$$

### Box-Muller 변환

$$
Z_0 = \sqrt{-2 \ln U_1} \cdot \cos(2\pi U_2)
$$

$$
Z_1 = \sqrt{-2 \ln U_1} \cdot \sin(2\pi U_2)
$$

### 뉴턴-랩슨 방법

$$
x_{n+1} = \frac{1}{2} \left( x_n + \frac{S}{x_n} \right)
$$

