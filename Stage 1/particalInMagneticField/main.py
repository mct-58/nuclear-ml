import numpy as np
import matplotlib.pyplot as plt

dt = 0.000000001

mass = 1.67*(10**-27)
charge = 1.6*(10**-19)

magneticField = np.array([0, 0, 0.05])
electricField = 0

initialPosition = np.array([0,0,0], dtype = float)
initialVelocity = np.array([1,0,0.1])

stateType = np.dtype([('position', 'f8', 3), ('velocity', 'f8', 3), ('acceleration', 'f8', 3), ('ke', 'f8')])

eulerResultsTemp = [] #List of tuples, of the form (position, velocity, acceleration, kinetic energy)
RK4ResultsTemp = []  #Same form

def lorentzAccel(position, velocity):
    F = charge * (electricField + np.cross(velocity, magneticField))
    return F/mass

def euler(timestep, latestDerivative):
    return latestDerivative * timestep

def RK4(timestep, position, velocity):
    k1Pos = velocity
    k1Vel = lorentzAccel(position, velocity)

    k2Pos = velocity + 0.5 * timestep * k1Vel
    k2Vel = lorentzAccel(position + 0.5 * timestep * k1Pos, k2Pos)

    k3Pos = velocity + 0.5 * timestep * k2Vel
    k3Vel = lorentzAccel(position + 0.5 * timestep * k2Pos, k3Pos)

    k4Pos = velocity + timestep * k3Vel
    k4Vel = lorentzAccel(position + timestep * k3Pos, k4Pos)

    newPosition = position + timestep * (1/6 * k1Pos + 1/3 * k2Pos + 1/3 * k3Pos + 1/6 * k4Pos)
    newVelocity = velocity + timestep * (1/6 * k1Vel + 1/3 * k2Vel + 1/3 * k3Vel + 1/6 * k4Vel)
    return newPosition, newVelocity

def kenergy(velocity):
    return 1/2 * mass * np.dot(velocity,velocity)

for i in range(10000):
    if i == 0:
        eulerResultsTemp.append((initialPosition, initialVelocity, lorentzAccel(initialPosition, initialVelocity), kenergy(initialVelocity)))
        RK4ResultsTemp.append((initialPosition, initialVelocity, lorentzAccel(initialPosition, initialVelocity), kenergy(initialVelocity)))
    else:
        velo = eulerResultsTemp[i-1][1] + euler(dt, eulerResultsTemp[i-1][2])
        pos = eulerResultsTemp[i-1][0] + euler(dt, eulerResultsTemp[i-1][1])
        eulerResultsTemp.append((pos, velo, lorentzAccel(pos, velo), kenergy(velo)))

        RK4Pos, RK4Velo = RK4(dt, RK4ResultsTemp[i-1][0], RK4ResultsTemp[i-1][1])
        RK4ResultsTemp.append((RK4Pos, RK4Velo, lorentzAccel(RK4Pos, RK4Velo), kenergy(RK4Velo)))


eulerResults = np.array(eulerResultsTemp, dtype=stateType)
RK4Results = np.array(RK4ResultsTemp, dtype=stateType)

def plotTrajectory(ax, states, label):
    ax.set_title(label + ', trajectory')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    if len(states) == 0:
        ax.text2D(0.5, 0.5, 'not implemented yet', ha='center', transform=ax.transAxes)
        return
    ax.plot(*states['position'].T)

def plotEnergy(ax, states, label):
    ax.set_title(label + ', kinetic energy')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('KE (J)')
    if len(states) == 0:
        ax.text(0.5, 0.5, 'not implemented yet', ha='center', transform=ax.transAxes)
        return
    ax.plot(np.arange(len(states)) * dt, states['ke'])

fig = plt.figure(figsize=(12, 9), constrained_layout=True)
fig.suptitle('Charged particle in a magnetic field')

plotTrajectory(fig.add_subplot(2, 2, 1, projection='3d'), eulerResults, 'euler')
plotTrajectory(fig.add_subplot(2, 2, 2, projection='3d'), RK4Results, 'RK4')
plotEnergy(fig.add_subplot(2, 2, 3), eulerResults, 'euler')
plotEnergy(fig.add_subplot(2, 2, 4), RK4Results, 'RK4')

plt.show()
