import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()

xTemp = np.linspace(0, 2 * np.pi, 200)
yTemp = np.sin(xTemp)
gNoise = rng.normal(0, 0.07, 200)

yFinal = yTemp + gNoise
xFinal = (xTemp - np.mean(xTemp)) / np.std(xTemp)

w = rng.normal(0, 1, 20)
b = np.zeros((20, 1))
v = rng.normal(0, 1, 20)
d = 0

eta = 0.01
loss = []

def preActivation(alpha):
    return np.outer(w, alpha) + b

def ReLU(beta):
    h = np.zeros(beta.shape)
    for i in range(beta.shape[1]):
        for j in range(beta.shape[0]):
            if beta[j][i] > 0:
                h[j][i] = beta[j][i]
    return h

def output(gamma):
    store = np.zeros(gamma.shape[1])
    for i in range(gamma.shape[1]):
        store[i] = np.dot(gamma[:, i], v) + d
    return store

def fPass(x):
    z = preActivation(x)
    h = ReLU(z)
    yHat = output(h)
    return z, h, yHat

def MSE(predictions, real):
    return sum(((real - predictions)**2) / len(real))

def backwards(z, h, yHat, y, x):
    e = -2 * (y - yHat) / len(y)
    pLpd = np.sum(e)
    pLpv = np.sum(e * h, 1)
    pLpb = np.reshape(np.sum(e * np.reshape(v , (len(v), 1)) * (z > 0), 1) , b.shape)
    pLpw = np.sum(e * np.reshape(v , (len(v), 1)) * (z > 0) * x, 1) 
    return pLpd, pLpv, pLpb, pLpw

def Ldiff(x, y, paramName, paramIndex):
    global w, b, v, d
    gw = w.copy()
    gb = b.copy()
    gv = v.copy()
    gd = float(d)
    tempD = {'w' : w, 'b' : b, 'v' : v}
    eps = 1e-5

    if paramName == 'd':
        d += eps
        Lplus = MSE(fPass(x)[2], y) 
        d-= 2 * eps
        Lminus = MSE(fPass(x)[2], y) 
    else:
        tempD[paramName][paramIndex] += eps
        Lplus = MSE(fPass(x)[2], y) 

        tempD[paramName][paramIndex] -= 2 * eps
        Lminus = MSE(fPass(x)[2], y)

    w = gw
    b = gb
    v = gv
    d = gd

    return (Lplus - Lminus) / (2*eps)


for _ in range(10000):
    z, h, yHat = fPass(xFinal)
    loss.append(MSE(yHat, yFinal))
    pLpd, pLpv, pLpb, pLpw = backwards(z, h, yHat, yFinal, xFinal)
    d -= pLpd * eta
    v -= pLpv * eta
    b -= pLpb * eta
    w -= pLpw * eta

plt.plot(loss)
plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()

plt.scatter(xFinal, yFinal, s=10, label='Data')
plt.plot(xFinal, fPass(xFinal)[2], color='orange', label='Network')
plt.xlabel('x (normalised)')
plt.ylabel('y')
plt.legend()
plt.show()

