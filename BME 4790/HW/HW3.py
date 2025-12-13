import matplotlib
import math
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

def SigmoidFunctionGenerator(x, noiseAmplitude, parameter):
    y = parameter[0]*np.exp(parameter[1]*(x-parameter[2]))/(parameter[2]+np.exp(parameter[1]*(x-parameter[2]))) + np.random.normal(0,noiseAmplitude,numSamples)
    return x,y

parameter = [100,20,0.5] # a determines max(sigmoid), b determines the slope, c determines the midpoint similar to the ld50
x = np.linspace(0,1,100)
noiseAmplitude = 5
numSamples = 200

x, y = SigmoidFunctionGenerator(noiseAmplitude, numSamples, parameter)

# Figure 1: Scatter plot of random linear data
plt.figure(1)  # Create Figure 1
plt.scatter(x, y, s=10)  # Set marker size to 10 for smaller dots
plt.xlabel('x')
plt.ylabel('y')
plt.title('Random Sigmoid Data with Noise')

plt.show()  # Non-blocking showFigure 1

# A

def TrainingDataGenerator(numSets, noiseAmplitude, numSamples):
    for i in range(numSets):
        parameter = [np.linspace(0,100,numSets)[i], np.linspace(10,50,numSets)[i], np.linspace(0,1,numSets)[i]]
        x, y = SigmoidFunctionGenerator(noiseAmplitude, numSamples, parameter)
        x = x/max(x)
        parameter = parameter / max(parameter)
    return x, parameter

numSets = 5
x, parameter = TrainingDataGenerator(numSets, noiseAmplitude, numSamples)