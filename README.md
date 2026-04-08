# PREDICTING MAP-READINESS WITH EKF-SLAM
Learn to predict whether a SLAM-based system is ready to act, using sequential features derived from its state estimates, uncertainty, and observations.

## Overview

Autonomous systems often rely on state estimation algorithms such as SLAM (Simultaneous Localization and Mapping) to operate in uncertain environments.

This project studies the following question:

> Given a stream of noisy observations and SLAM estimates over random-walk, at what point has the agent explored enough of its envi-ronment to reliably navigate it and execute a given task?

We simulate a robot operating a random exploration policy using noisy landmark observations and EKF-SLAM to (1) predict whether the map is ready at a given time and (2) estimate the remaining time until map-readiness.

---

## Objective

We define a binary label at each time step:

$$
Y_t =
\begin{cases}
1 & \text{if acting at time } t \text{ would succeed}, \\\\
0 & \text{otherwise}
\end{cases}
$$

The goal is to learn

$$
\mathbb{P}(Y_t = 1 \mid \text{information available at time } t).
$$

This transforms SLAM into a **decision-making problem under uncertainty**, rather than just an estimation problem.

---

## Key Idea

At each time step, the system has access to:
- an estimated state $\hat{x}_t$,
- an uncertainty measure $P_t$,
- a set of observations,
- recent control inputs.

However, uncertainty alone is not always reliable. A system may be:
- **overconfident** (low covariance, high error), or
- **underconfident** (high covariance, low error).

This project investigates whether we can **predict true readiness-to-act better than simple uncertainty thresholds**.

---

## Simulation Framework

We generate a sequential dataset using a closed-loop robot simulation.

### Environment
- 2D world with randomly placed landmarks
- Randomly generated target location
- Limited sensor range and field of view

### Robot Dynamics

State:
$$
x_t = [p_x, p_y, \theta]
$$

Control:
$$
u_t = [v_t, \omega_t]
$$

Dynamics:
$$
x_{t+1} = f(x_t, u_t) + w_t
$$

### Observations

For each landmark $j$,
$$
z_{t,j} =
\begin{bmatrix}
r_{t,j} \\\\
\phi_{t,j}
\end{bmatrix}
=
h(x_t, m_j) + v_{t,j}
$$

where:
- $r$ is range,
- $\phi$ is bearing.

### Estimation

We use EKF-SLAM to jointly estimate:
- robot pose,
- landmark positions.

### Control Policy

Controls are computed using the **estimated state**:

$$
u_t = \pi(\hat{x}_t, \text{goal})
$$

This creates a fully **closed-loop system**.

---

## Data Generation Process

Each episode proceeds as follows:

1. Initialize world, robot state, and SLAM.
2. For each time step $t$:
   - compute control $u_t$ from $\hat{x}_t$,
   - propagate true state $x_t$,
   - generate observations $z_t$,
   - update SLAM estimate $(\hat{x}_t, P_t)$,
   - compute label $Y_t$.
3. Stop when:
   - the system decides to act, or
   - the maximum number of steps is reached.

---

## Dataset Description

Each row corresponds to one time step.

### True State
- `true_x`, `true_y`, `true_theta`

### Estimated State
- `est_x`, `est_y`, `est_theta`

### Controls
- `u_v`, `u_w`

### Observations
- `n_obs`

### Uncertainty
- `trace_pose_cov`
- `pos_uncertainty`

### Performance Metrics
- `loc_error`
- `true_dist_to_goal`
- `est_dist_to_goal`

### Decision Variables
- `act_now` — baseline decision rule
- `success_if_act_now` — ground-truth label $Y_t$

---

## Learning Tasks

### 1. Readiness Prediction

Estimate
$$
\mathbb{P}(Y_t = 1)
$$

using:
- regression,
- random forests,
- time-series features.

### 2. Decision Strategy Comparison

Compare:
- threshold-based rules,
- learned models.

Evaluate:
- success rate,
- false action rate,
- time-to-decision.

### 3. Sequential Analysis

Study:
- evolution of uncertainty $P_t$,
- relationship between observations and convergence,
- failure modes such as drift and overconfidence.

---

## Evaluation Metrics

- Accuracy / ROC-AUC
- Precision / Recall
- Time-to-decision
- Success rate
- Calibration: compare uncertainty vs. true error

---

## Code Structure

`ekf_slam_simulator.py`
- simulation configuration
- world generation
- robot dynamics
- sensor model
- EKF-SLAM
- control policy
- episode simulation
- data export

---

## Outputs

`sim_output/`
- `time_series.csv`
- `episode_summary.csv`
- `config.json`

---

## How to Run

```bash
python ekf_slam_simulator.py
