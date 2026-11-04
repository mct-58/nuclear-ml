import numpy as np
import matplotlib.pyplot as plt

#lambda in units/day
isotopes = {'U235' : {'lambda' : np.log(2)/(7.04e8 * 365), 'N' : [6.02e23]} ,
         'Th231' : {'lambda' : np.log(2)/(25.5/24), 'N' : [0]},
          'Pa231' : {'lambda' : np.log(2)/(3.28e4 * 365), 'N' : [0]}
         }

lmda = np.array([isotopes['U235']['lambda'], isotopes['Th231']['lambda'], isotopes['Pa231']['lambda']])
times = [0]

def RK4(currentState, dt, derivative):
    k1 = derivative(currentState)
    k2 = derivative(currentState + dt/2 * k1)
    k3 = derivative(currentState + dt/2 * k2)
    k4 = derivative(currentState + dt * k3)
    return currentState + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def decayRate(N):
    temp = np.concatenate(([0], lmda * N))
    return temp[:-1] - (lmda * N)

dt = 0.1
tolerance = 1e-6
endTime = 1e7

for i in range(50000000):
    if times[-1] < endTime:
        current = np.array([isotopes['U235']['N'][-1], isotopes['Th231']['N'][-1], isotopes['Pa231']['N'][-1]])

        oneBigStep = RK4(current, dt, decayRate)
        halfStep = RK4(current, dt/2, decayRate)
        twoSmallSteps = RK4(halfStep, dt/2, decayRate)

        error = np.max(np.abs(twoSmallSteps - oneBigStep) / twoSmallSteps)

        if error < tolerance:
            times.append(times[-1] + dt)
            isotopes['U235']['N'].append(twoSmallSteps[0])
            isotopes['Th231']['N'].append(twoSmallSteps[1])
            isotopes['Pa231']['N'].append(twoSmallSteps[2])
            dt = dt * 2
        else:
            dt = dt / 2

plt.plot(times, isotopes['Th231']['N'], label="Th231")
plt.plot(times, isotopes['Pa231']['N'], label="Pa231")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
