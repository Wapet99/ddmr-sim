# DDMR Simulation — Python Port

A Python port of a MATLAB simulation for a **Differential Drive Mobile Robot (DDMR)** operating in a structured indoor environment (a Mars habitat). The robot navigates autonomously using polynomial spline trajectory planning, Model Predictive Control (MPC), and optional Kalman Filter state estimation.

---

## Overview

The simulation models a two-wheeled DDMR with realistic actuator dynamics, rigid-body mechanics, and nonholonomic constraints. Two operational scenarios are supported:

- **`track`** — The robot tracks a pre-planned polynomial spline trajectory through a series of waypoints, minimising position and heading error.
- **`dock`** — The robot navigates to a fixed docking/charging station, subject to obstacle and wall constraints enforced via nonlinear MPC.

---

## Features

- **Rigid-body dynamics** derived from first principles, including actuator torque mapping and a reduced-order momentum state representation
- **Runge-Kutta 3/8 integration** for accurate state propagation within each MPC prediction step
- **Gradient-augmented optimisation** — analytical Jacobians propagated alongside state trajectories to accelerate the MPC solver
- **Polynomial spline trajectory design** — cubic/quintic splines fitted through user-defined waypoints with continuous velocity boundary conditions
- **Two MPC formulations:**
  - `track`: nonlinear least-squares (`scipy.optimize.least_squares`) with optional analytical Jacobian
  - `dock`: nonlinear constrained optimisation (`scipy.optimize.minimize`, SLSQP) with wall and obstacle constraints
- **Kalman Filter state estimator** (scaffolded; toggled via `runStateEstimator` / `useStateEstimate` flags in `RunMain.py`)
- **Animated visualisation** of the robot's trajectory overlaid on a grid map of the habitat

---

## Project Structure

```
├── RunMain.py                  # Entry point — simulation loop, plotting, animation
├── robotParameters.py          # Robot physical and control parameter container
├── controlParameters.py        # MPC horizon lengths, cost weights, actuator bounds
├── robotInitialConditions.py   # Scenario-specific initial state vectors
├── robotDynamics.py            # Nonlinear state-space dynamics + analytical Jacobians
├── robotDynamicsAdapter.py     # Wrapper for scipy.integrate.solve_ivp compatibility
├── rigidBodyMassMatrix.py      # 6-DOF rigid-body mass matrix (Euler-Newton)
├── robotConstraints.py         # Nonholonomic constraint matrices (Bc, Bcp)
├── actuatorConfiguration.py    # Actuator force/torque mapping matrix (Ba)
├── skew.py                     # Skew-symmetric matrix utility
├── trajectoryDesign.py         # Waypoint definition and spline coefficient generation
├── trajectoryEval.py           # Spline evaluation at arbitrary query times
├── controlMPC.py               # MPC solver (scenario-switching, warm-starting)
├── errorMPC.py                 # Tracking error residuals + Jacobians for least-squares MPC
├── costMPC.py                  # Scalar cost + gradient for docking MPC
├── nonlconMPC.py               # Nonlinear inequality constraints (walls, obstacles) for docking
└── augmentGradients.py         # Augmented ODE for propagating state-input Jacobians
```

---

## Getting Started

### Requirements

```
python >= 3.10
numpy
scipy
matplotlib
```

Install dependencies:

```bash
pip install numpy scipy matplotlib
```

### Running the Simulation

```bash
python RunMain.py
```

Key flags at the top of `RunMain.py`:

| Flag | Description |
|---|---|
| `scenario` | `'track'` or `'dock'` |
| `controller` | `'MPC'` (only option currently wired up) |
| `useGradientOpt` | `True` to use analytical Jacobians in the MPC solver (faster) |
| `runStateEstimator` | `True` to run the Kalman Filter update step |
| `useStateEstimate` | `True` to feed the state estimate into the controller |
| `dt` | Simulation timestep (seconds) |
| `T` | Total simulation duration (seconds) |

### Required Data File

The animation requires `map.mat` (a MATLAB `.mat` file containing `map`, `Egridlines`, and `Ngridlines`) to be present in the working directory.

---

## System Model

### States

The robot uses a **reduced-order momentum state** representation with 5 states:

| Index | Symbol | Description |
|---|---|---|
| 0 | `p1` | Generalised momentum component 1 |
| 1 | `p2` | Generalised momentum component 2 |
| 2 | `N` | North position [m] |
| 3 | `E` | East position [m] |
| 4 | `ψ` | Yaw angle [rad] |

### Inputs

Two scalar motor voltages `u = [u_L, u_R]` (bounded to ±6 V by default).

### Key Physical Parameters (defaults)

| Parameter | Value | Description |
|---|---|---|
| `m` | 3 kg | Robot mass |
| `Iz` | 0.0675 kg·m² | Yaw inertia |
| `l` | 0.2 m | Longitudinal distance B → wheels |
| `d` | 0.3 m | Track width (wheel separation) |
| `c` | 0.1 m | Longitudinal distance B → geometric centre C |
| `Km` | 0.014 N·m/A | Motor constant |
| `Ra` | 0.657 Ω | Armature resistance |
| `Ng` | 5 | Gear ratio |
| `rw` | 0.04 m | Wheel radius |
| `g` | 3.71 m/s² | Mars surface gravity |

---

## MPC Formulation

### Tracking (`track`)

Minimises a weighted sum of squared errors over a prediction horizon `Np`:

```
min  Σ [ qr·‖rC - rP‖² + qψ·(ψ - ψ*)² + ru·‖u‖² ]
 U
```

where `rC` is the robot's geometric centre, `rP` is the spline reference position, and `ψ*` is the desired heading (tangent to the spline). Solved via `scipy.optimize.least_squares` with optional analytical Jacobian supply.

### Docking (`dock`)

Minimises a terminal position and heading cost subject to nonlinear inequality constraints encoding wall boundaries and circular obstacles. Solved via `scipy.optimize.minimize` (SLSQP).

### Default Cost Weights

| Parameter | Value | Description |
|---|---|---|
| `qr` | 10000 | Position error penalty |
| `qpsi` | 10 | Heading error penalty |
| `ru` | 0.1 | Actuator effort penalty |

---

## Trajectory Design

Trajectories are defined as a sequence of **waypoints** — each specifying position and velocity in North/East coordinates at a given time — in `trajectoryDesign.py`. A polynomial spline of order `2·(derivative_order) - 1` is fitted through consecutive waypoints by solving a linear system for the spline coefficients. The resulting trajectory is evaluated at query times by `trajectoryEval.py`.

---

## Notes

- The Kalman Filter estimator is scaffolded (the `simulateMeasurement()` call and EKF update are placeholders) and not active by default.
- The `dock` scenario's `nonlcon` constraint is currently wired as an equality constraint in `controlMPC.py` — this should be `'ineq'` for a standard inequality constraint formulation; review before use.
- Coordinate convention throughout is **NED-style** (North–East) for 2D position, consistent with the original MATLAB codebase.
