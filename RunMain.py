import numpy as np
import scipy.io
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load habitat map
map_data = scipy.io.loadmat('map.mat')
map_grid = map_data['map'].astype(int)
Egridlines = map_data['Egridlines'].flatten()
Ngridlines = map_data['Ngridlines'].flatten()

# Import functions
from robotParameters import RobotParameters
from robotDynamicsAdapter import robotDynamicsAdapter
from robotInitialConditions import robotInitialConditions
from controlMPC import controlMPC
from trajectoryDesign import trajectoryDesign

# Simulation parameters
scenario = 'track'
controller = 'MPC'
runStateEstimator = False
useStateEstimate = False
useGradientOpt = True
match scenario:
    case 'track':
        dt = 0.1
        T = 30
    case 'dock':
        dt = 0.1
        T = 6
    case _:
        print("Unknown Scenario")

# Set initial conditions
param = RobotParameters(scenario=scenario, controller=controller, dt=dt, T=T, useGO = useGradientOpt)
x0 = robotInitialConditions(param)
u = np.array([0, 0]).reshape(2,1)
if param.scenario == 'track':
    param.traj = trajectoryDesign(param)

# Run simulation
tHist = np.arange(0, param.T + param.dt, param.dt)
xHist = np.zeros((5, len(tHist)))
xHist[:, 0] = x0
U = []
uHist = np.zeros((2, len(tHist)))
uHist[:, 0] = u.flatten()
muHist = np.zeros((5, len(tHist)))
SHist = np.zeros((5, 5, len(tHist)))


for k in range(len(tHist) - 1):

    sol = scipy.integrate.solve_ivp(robotDynamicsAdapter, [tHist[k], tHist[k+1]], xHist[:, k], args=(u, param), method='RK45')
    xHist[:, k+1] = sol.y[:, -1]

    if runStateEstimator:
        # Take sensor measurements
        yacc = simulateMeasurement()

    if useStateEstimate and runStateEstimator:
        xctrl = muHist[0:5, k+1]
    else:
        xctrl = xHist[0:5, k+1]
    
    #Compute next control action
    U = controlMPC(tHist[k+1], xctrl, u, U, param)
    #u = np.array([0.4, 0.4]).reshape(2,1)  # Placeholder control action
    u = U[:2].reshape(2,1)
    uHist[:, k+1] = u.flatten()

## Plot results
NHist = xHist[2, :]
EHist = xHist[3, :]
psiHist = xHist[4, :]

fig, axs = plt.subplots(3, 3, figsize=(12, 8))
axs[0, 0].plot(tHist, NHist)
axs[0, 0].set_title('North position [m]')
axs[1, 0].plot(tHist, EHist)
axs[1, 0].set_title('East position [m]')
axs[2, 0].plot(tHist, psiHist * 180 / np.pi) # convert from radians to degrees
axs[2, 0].set_title('Yaw angle [deg]')

plt.tight_layout()
plt.show()

## Animation
Egrid, Ngrid = np.meshgrid(Egridlines, Ngridlines)
Cgrid = np.zeros_like(Egrid)
Cgrid[:-1, :-1] = ~map_grid  # Using your loaded map data

# Initialize figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(Egridlines[0], Egridlines[-1])
ax.set_ylim(Ngridlines[0], Ngridlines[-1])
ax.set_xlabel("East [m]")
ax.set_ylabel("North [m]")

# Plot habitat map using the loaded data
ax.pcolormesh(Egrid, Ngrid, Cgrid, cmap="gray", shading="auto")

# Objects in the environment
hDust = plt.Polygon([[12.5, 0.5], [13, 0.5], [13, 1], [12.5, 1]], color="red")
hCharge = plt.Polygon([[3.05, 0.5], [3.55, 0.5], [3.55, 1.0], [3.05, 1.0]], color="red")

ax.add_patch(hDust)
ax.add_patch(hCharge)

# Dynamic objects (robot trajectory)
#hO, = ax.plot([], [], 'k-', lw=2)  # Robot outline
#outline = plt.Circle((0,0), radius=0.2, fill=False, edgecolor='k', lw=1)
#ax.add_patch(outline)
hL, = ax.plot([], [], 'ro', markersize=1)  # Left wheel
hR, = ax.plot([], [], 'go', markersize=1)  # Right wheel
hS, = ax.plot([], [], 'b1', markersize=1)  # Sensor
hB, = ax.plot([], [], 'ko', markersize=1)  # Base
hC, = ax.plot([], [], 'bo', markersize=1)  # Centre

# Animation update function
def update(frame):
    Rnb = np.array([[np.cos(psiHist[frame]), -np.sin(psiHist[frame]), 0],
                    [np.sin(psiHist[frame]),  np.cos(psiHist[frame]), 0],
                    [0, 0, 1]])  # Rotation matrix

    rBNn = np.array([NHist[frame], EHist[frame], 0])
    rCNn = rBNn + (Rnb @ param.rCBb).flatten()
    rLNn = rBNn + (Rnb @ param.rLBb).flatten()
    rRNn = rBNn + (Rnb @ param.rRBb).flatten()
    rSNn = rBNn + (Rnb @ param.rSBb).flatten()

    # Update object positions
    hC.set_data([rCNn[1]], [rCNn[0]])
    hL.set_data([rLNn[1]], [rLNn[0]])
    hR.set_data([rRNn[1]], [rRNn[0]])
    hS.set_data([rSNn[1]], [rSNn[0]])
    hB.set_data([rBNn[1]], [rBNn[0]])
    #outline.set_center((float(rCNn[1]), float(rCNn[0])))

    return hC, hL, hR, hS, hB#, outline

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(tHist), interval=param.dt * 1000, blit=True)
plt.show()