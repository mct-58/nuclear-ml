import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(67)

def realLine(x):
    m = 2.6
    c = 3
    return m * x + c

mTracker = 0.5
cTracker = 5

xVals = np.linspace(0, 20, 200) - 10
YVals = realLine(xVals)

noise = rng.normal(0, 1.7, 200)

fakeYVals = YVals + noise

def f(mTracker, cTracker, x):
    return mTracker * x + cTracker

def MSE(guessedY, realY):
    differencesSquared = (guessedY - realY) ** 2
    return np.sum(differencesSquared) / len(realY)

def derWRTm(m, c, x, y):
    sum = 0
    for i in range(len(x)):
        sum += m * (x[i])**2 + c * x[i] - x[i] * y[i]
    
    return 2/(len(x)) * sum

def derWRTc(m, c, x, y):
    sum = 0
    for i in range(len(x)):
        sum += m * x[i] + c - y[i]
    
    return 2/(len(x)) * sum

errorList = []
mlearningRate = 0.0001
clearningRate = 0.00745

for i in range(1000):
    guesses = f(mTracker, cTracker, xVals)
    errorList.append(MSE(guesses, fakeYVals))
    mDer = derWRTm(mTracker, cTracker, xVals, fakeYVals)
    cDer = derWRTc(mTracker, cTracker, xVals, fakeYVals)
    mTracker -= mDer * mlearningRate
    cTracker -= cDer * clearningRate

print(f'Predicted gradient = {mTracker} and the intersect is {cTracker}')

plt.plot(errorList)
plt.yscale('log')
plt.xlabel("iteration")
plt.ylabel("MSE")
plt.show()