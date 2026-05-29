# PREDICTING MAP-READINESS WITH EKF-SLAM  
Learn to predict whether a SLAM-based system is ready to act, using sequential features derived from its state estimates, uncertainty, and observations.

## Overview

Autonomous systems often rely on state estimation algorithms such as SLAM (Simultaneous Localization and Mapping) to operate in uncertain environments.

This project studies the following question:

> Given a stream of noisy observations and SLAM estimates under random exploration, at what point has the agent explored enough of its environment to reliably navigate it and execute a given task?

We simulate a robot operating under a random exploration policy using noisy landmark observations and EKF-SLAM to:
1. predict whether the map is ready at a given time, and  
2. estimate the remaining time until map-readiness.

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
