import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math

def GenerateRandomLinearData(range,parameter,noiseAmplitude,numSamples):
    x = np.linspace(range[0],range[1],numSamples)
    y = parameter[0]*x + parameter[1] + np.random.uniform(-noiseAmplitude,noiseAmplitude,numSamples)
    return x,y

range = [0,10]
parameter = [2,1]
noiseAmplitude = 1
numSamples = 100

x,y = GenerateRandomLinearData(range,parameter,noiseAmplitude,numSamples)
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Random Linear Data with Noise')
plt.show()

def SumOfSquares(x,y,parameter):
    y_pred = parameter[0]*x + parameter[1]
    return np.sum((y - y_pred)**2)

def SumOfSquaresGradient(x,y,mRange,bRange,numPoints):
    m = np.linspace(mRange[0],mRange[1],numPoints)
    b = np.linspace(bRange[0],bRange[1],numPoints)
    M,B = np.meshgrid(m,b)
    Gradient = np.zeros(M.shape)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Gradient(i,j) = SumOfSquares(x,y,[M[i,j],B[i,j]])
    return Gradient

mRange = [-5,5]
bRange = [-5,5]
numPoints = 100
Gradient = SumOfSquaresGradient(x,y,mRange,bRange,numPoints)
np.contour(mRange,bRange,Gradient,500,'jet')

