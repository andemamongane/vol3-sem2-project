# PREDICTING MAP-READINESS WITH EKF-SLAM  
Learn to predict whether a SLAM-based system is ready to act, using sequential features derived from its state estimates, uncertainty, and observations.

## Overview

Autonomous systems often rely on state estimation algorithms such as SLAM (Simultaneous Localization and Mapping) to operate in uncertain environments.

This project studies the following question:

> Given a stream of noisy observations and SLAM estimates under random exploration, at what point has the agent explored enough of its environment to reliably navigate it and execute a given task?

We simulate a robot operating under a random exploration policy using noisy landmark observations and EKF-SLAM to:
1. predict whether the map is ready at a given time, and  
2. estimate the remaining time until map-readiness.

---

## EKF-SLAM

We use **Extended Kalman Filter SLAM (EKF-SLAM)** to recursively estimate the joint robot-map state over time. At each step, the filter maintains

- a mean estimate \(\hat{\mathbf{x}}_t\),
- and a covariance matrix \(\Sigma_t\).

Each iteration consists of:

1. **Prediction**: propagate the state estimate and covariance through the motion model,  
2. **Update**: incorporate noisy landmark observations using a local linearization of the nonlinear observation model.

This produces a time series of state estimates, uncertainties, and innovation statistics that can be used to assess map quality.

---

## What is Map-Readiness?

The central question of this project is:

> **When has the robot explored enough of the environment that its map is reliable enough to support downstream action?**

In simulation, we define map-readiness using access to ground truth. A map may be declared **ready** when one or more criteria are satisfied, such as:

- map RMSE falls below a threshold,
- robot pose uncertainty becomes sufficiently small,
- enough landmarks have been observed and refined,
- or the system achieves stable estimation performance over time.

This gives a binary readiness label

\[
y_t =
\begin{cases}
1, & \text{if the map is ready at time } t, \\
0, & \text{otherwise.}
\end{cases}
\]

---

## Feature Engineering

From the EKF-SLAM outputs, we extract sequential features that summarize the current quality and maturity of the map.

### State and Uncertainty Features
- trace or determinant of the covariance matrix,
- robot-pose uncertainty,
- landmark uncertainty,
- leading covariance eigenvalues.

### Observation Features
- number of landmarks observed at each step,
- number of newly discovered landmarks,
- revisit counts,
- observation frequency over rolling windows.

### Consistency and Innovation Features
- innovation norm,
- innovation variance,
- covariance shrinkage rate,
- stability of recent updates.

### Temporal Features
- rolling averages,
- slope or trend features,
- recent-vs-past comparison windows.

These features are used to train models that predict readiness directly from the SLAM trajectory.

---

## Learning Tasks

We consider two supervised learning tasks.

### 1. Map-Readiness Classification

Predict whether the map is ready at time \(t\):

\[
P(y_t = 1 \mid \text{features up to time } t).
\]

### 2. Time-to-Readiness Regression

Estimate the remaining number of steps until the map becomes ready:

\[
\tau_t = \text{remaining steps until readiness}.
\]

Together, these tasks help determine both **whether** the system is ready and **how long remains** if it is not.

---

## Models

Possible baseline models include:

- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- Simple temporal models on rolling features

More advanced extensions may include:

- Gaussian Process regression,
- sequence models,
- or uncertainty-aware meta-estimators.

---

## Evaluation

We evaluate the learned predictors using both machine learning metrics and SLAM-quality metrics.

### Classification Metrics
- Accuracy  
- Precision / Recall  
- F1-score  
- ROC-AUC  

### Regression Metrics
- RMSE  
- MAE  

### SLAM / Ground-Truth Metrics
- map RMSE,
- pose error,
- landmark coverage,
- covariance consistency.

---

## Project Pipeline

The full pipeline is:

1. **Simulate** random exploration trajectories in a 2D landmark environment  
2. **Run EKF-SLAM** on noisy observations  
3. **Extract sequential features** from filter estimates and observations  
4. **Label readiness** using ground-truth map quality criteria  
5. **Train predictors** for readiness and time-to-readiness  
6. **Evaluate** how accurately the learned models detect useful stopping points for exploration

---

## Repository Structure

```text
project/
│
├── data/                # simulated trajectories and processed datasets
├── slam/                # EKF-SLAM implementation
├── features/            # feature extraction and preprocessing
├── models/              # classification and regression models
├── evaluation/          # metrics, plots, and experiments
├── notebooks/           # exploratory analysis and prototyping
└── README.md
