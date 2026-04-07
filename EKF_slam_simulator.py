from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================


@dataclass
class SimConfig:
    dt: float = 0.1
    max_steps: int = 500
    world_size: float = 20.0
    n_landmarks: int = 15
    sensor_range: float = 8.0
    fov_rad: float = math.pi  # 180 degrees
    motion_noise_std: Tuple[float, float, float] = (0.03, 0.03, 0.01)
    meas_noise_std: Tuple[float, float] = (0.15, 0.03)  # range, bearing
    process_noise_diag: Tuple[float, float, float] = (0.02, 0.02, 0.01)
    init_pose_std: Tuple[float, float, float] = (0.2, 0.2, 0.05)
    goal_tolerance_true: float = 0.5
    act_est_tolerance: float = 0.6
    act_uncertainty_threshold: float = 0.35
    kp_v: float = 1.0
    kp_w: float = 2.5
    max_v: float = 1.2
    max_w: float = 1.5
    stop_v_near_goal: float = 0.25
    seed: int = 42


# ============================================================
# Helpers
# ============================================================


def wrap_angle(theta: float) -> float:
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ============================================================
# World generation
# ============================================================


@dataclass
class World:
    landmarks: np.ndarray  # shape (N, 2)
    goal: np.ndarray       # shape (2,)


def make_world(cfg: SimConfig, rng: np.random.Generator) -> World:
    margin = 2.0
    landmarks = rng.uniform(
        low=-cfg.world_size / 2 + margin,
        high=cfg.world_size / 2 - margin,
        size=(cfg.n_landmarks, 2),
    )
    goal = rng.uniform(
        low=-cfg.world_size / 2 + margin,
        high=cfg.world_size / 2 - margin,
        size=(2,),
    )
    return World(landmarks=landmarks, goal=goal)


# ============================================================
# True robot dynamics and sensing
# ============================================================


def motion_model(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    State x = [px, py, theta]
    Control u = [v, w]
    """
    px, py, th = x
    v, w = u
    nx = np.array([
        px + v * np.cos(th) * dt,
        py + v * np.sin(th) * dt,
        wrap_angle(th + w * dt),
    ])
    return nx


def simulate_true_motion(
    x_true: np.ndarray,
    u: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    x_next = motion_model(x_true, u, cfg.dt)
    noise = rng.normal(0.0, cfg.motion_noise_std, size=3)
    x_next = x_next + noise
    x_next[2] = wrap_angle(x_next[2])
    return x_next


def observe_landmarks(
    x_true: np.ndarray,
    world: World,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    observations: List[Dict[str, float]] = []
    px, py, th = x_true

    for j, lm in enumerate(world.landmarks):
        dx = lm[0] - px
        dy = lm[1] - py
        r = np.hypot(dx, dy)
        if r > cfg.sensor_range:
            continue

        bearing = wrap_angle(np.arctan2(dy, dx) - th)
        if abs(bearing) > cfg.fov_rad / 2:
            continue

        noisy_r = r + rng.normal(0.0, cfg.meas_noise_std[0])
        noisy_b = wrap_angle(bearing + rng.normal(0.0, cfg.meas_noise_std[1]))
        observations.append({"id": j, "range": noisy_r, "bearing": noisy_b})

    return observations


# ============================================================
# EKF-SLAM implementation
# ============================================================


class EKFSLAM:
    def __init__(self, cfg: SimConfig, n_landmarks: int):
        self.cfg = cfg
        self.n_landmarks = n_landmarks
        self.state_dim = 3 + 2 * n_landmarks

        self.mu = np.zeros(self.state_dim)
        self.Sigma = np.eye(self.state_dim) * 1e6
        self.Sigma[:3, :3] = np.diag(np.square(cfg.init_pose_std))

        self.observed = np.zeros(n_landmarks, dtype=bool)

        self.Q = np.diag(np.square(cfg.process_noise_diag))
        self.R = np.diag(np.square(cfg.meas_noise_std))

    def pose(self) -> np.ndarray:
        return self.mu[:3].copy()

    def pose_cov(self) -> np.ndarray:
        return self.Sigma[:3, :3].copy()

    def predict(self, u: np.ndarray):
        px, py, th = self.mu[:3]
        v, w = u
        dt = self.cfg.dt

        # Predict mean
        self.mu[:3] = motion_model(self.mu[:3], u, dt)

        # Jacobian wrt robot pose
        Gx = np.array([
            [1.0, 0.0, -v * np.sin(th) * dt],
            [0.0, 1.0,  v * np.cos(th) * dt],
            [0.0, 0.0, 1.0],
        ])

        G = np.eye(self.state_dim)
        G[:3, :3] = Gx

        Fx = np.zeros((3, self.state_dim))
        Fx[:3, :3] = np.eye(3)

        self.Sigma = G @ self.Sigma @ G.T + Fx.T @ self.Q @ Fx
        self.mu[2] = wrap_angle(self.mu[2])

    def _landmark_slice(self, lm_id: int) -> slice:
        start = 3 + 2 * lm_id
        return slice(start, start + 2)

    def _initialize_landmark(self, obs: Dict[str, float]):
        lm_id = int(obs["id"])
        r = float(obs["range"])
        b = float(obs["bearing"])
        px, py, th = self.mu[:3]

        lx = px + r * np.cos(th + b)
        ly = py + r * np.sin(th + b)
        sl = self._landmark_slice(lm_id)
        self.mu[sl] = np.array([lx, ly])
        self.observed[lm_id] = True

        # A moderate initial covariance for newly seen landmarks
        self.Sigma[sl, sl] = np.diag([1.0, 1.0])
        self.Sigma[sl, :3] = 0.0
        self.Sigma[:3, sl] = 0.0

    def update(self, observations: List[Dict[str, float]]):
        for obs in observations:
            lm_id = int(obs["id"])
            if not self.observed[lm_id]:
                self._initialize_landmark(obs)

            sl = self._landmark_slice(lm_id)
            lx, ly = self.mu[sl]
            px, py, th = self.mu[:3]

            dx = lx - px
            dy = ly - py
            q = dx * dx + dy * dy
            if q < 1e-12:
                continue

            sqrt_q = np.sqrt(q)
            z_hat = np.array([
                sqrt_q,
                wrap_angle(np.arctan2(dy, dx) - th),
            ])
            z = np.array([obs["range"], obs["bearing"]])
            y = z - z_hat
            y[1] = wrap_angle(y[1])

            H = np.zeros((2, self.state_dim))
            # Robot pose block
            H[:, :3] = np.array([
                [-dx / sqrt_q, -dy / sqrt_q, 0.0],
                [dy / q,      -dx / q,      -1.0],
            ])
            # Landmark block
            H[:, sl] = np.array([
                [dx / sqrt_q, dy / sqrt_q],
                [-dy / q,      dx / q],
            ])

            S = H @ self.Sigma @ H.T + self.R
            K = self.Sigma @ H.T @ np.linalg.inv(S)

            self.mu = self.mu + K @ y
            self.mu[2] = wrap_angle(self.mu[2])
            I = np.eye(self.state_dim)
            self.Sigma = (I - K @ H) @ self.Sigma


# ============================================================
# Control policy and action decision
# ============================================================


def goal_controller(x_est: np.ndarray, goal: np.ndarray, cfg: SimConfig) -> np.ndarray:
    px, py, th = x_est
    dx = goal[0] - px
    dy = goal[1] - py
    dist = np.hypot(dx, dy)
    desired_heading = np.arctan2(dy, dx)
    heading_error = wrap_angle(desired_heading - th)

    v = cfg.kp_v * dist
    w = cfg.kp_w * heading_error

    if dist < 1.0:
        v = min(v, cfg.stop_v_near_goal)

    v = clip(v, 0.0, cfg.max_v)
    w = clip(w, -cfg.max_w, cfg.max_w)
    return np.array([v, w])


def should_act_now(x_est: np.ndarray, pose_cov: np.ndarray, goal: np.ndarray, cfg: SimConfig) -> bool:
    est_dist = np.linalg.norm(x_est[:2] - goal)
    pos_unc = float(np.sqrt(np.trace(pose_cov[:2, :2])))
    return (est_dist < cfg.act_est_tolerance) and (pos_unc < cfg.act_uncertainty_threshold)


def action_would_succeed(x_true: np.ndarray, goal: np.ndarray, cfg: SimConfig) -> bool:
    true_dist = np.linalg.norm(x_true[:2] - goal)
    return true_dist < cfg.goal_tolerance_true


# ============================================================
# Episode simulation
# ============================================================


def simulate_episode(
    episode_id: int,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    world = make_world(cfg, rng)

    # True state starts at origin
    x_true = np.array([0.0, 0.0, 0.0])

    # EKF-SLAM estimate starts near origin with noise
    x0_noise = rng.normal(0.0, cfg.init_pose_std, size=3)
    ekf = EKFSLAM(cfg, cfg.n_landmarks)
    ekf.mu[:3] = np.array([0.0, 0.0, 0.0]) + x0_noise
    ekf.mu[2] = wrap_angle(ekf.mu[2])

    rows: List[Dict[str, float]] = []
    done_reason = "max_steps"

    for t in range(cfg.max_steps):
        x_est = ekf.pose()
        P_pose = ekf.pose_cov()

        # Policy depends on estimated pose, not true pose.
        u = goal_controller(x_est, world.goal, cfg)

        # Step true system
        x_true = simulate_true_motion(x_true, u, cfg, rng)

        # Generate observations from truth
        obs = observe_landmarks(x_true, world, cfg, rng)

        # EKF-SLAM update
        ekf.predict(u)
        ekf.update(obs)

        x_est_new = ekf.pose()
        P_pose_new = ekf.pose_cov()

        loc_err = float(np.linalg.norm(x_true[:2] - x_est_new[:2]))
        true_dist = float(np.linalg.norm(x_true[:2] - world.goal))
        est_dist = float(np.linalg.norm(x_est_new[:2] - world.goal))
        pos_unc = float(np.sqrt(np.trace(P_pose_new[:2, :2])))
        act_now = should_act_now(x_est_new, P_pose_new, world.goal, cfg)
        succeed_now = action_would_succeed(x_true, world.goal, cfg)

        row = {
            "episode_id": episode_id,
            "t": t,
            "dt": cfg.dt,
            "true_x": x_true[0],
            "true_y": x_true[1],
            "true_theta": x_true[2],
            "est_x": x_est_new[0],
            "est_y": x_est_new[1],
            "est_theta": x_est_new[2],
            "goal_x": world.goal[0],
            "goal_y": world.goal[1],
            "u_v": u[0],
            "u_w": u[1],
            "n_obs": len(obs),
            "trace_pose_cov": float(np.trace(P_pose_new)),
            "pos_uncertainty": pos_unc,
            "loc_error": loc_err,
            "true_dist_to_goal": true_dist,
            "est_dist_to_goal": est_dist,
            "act_now": int(act_now),
            "success_if_act_now": int(succeed_now),
        }
        rows.append(row)

        if act_now:
            done_reason = "acted_success" if succeed_now else "acted_failure"
            break

    # Add done reason to all rows in the episode for convenience.
    for row in rows:
        row["done_reason"] = done_reason

    return pd.DataFrame(rows)


# ============================================================
# Batch runner and storage
# ============================================================


def run_experiment(
    n_episodes: int = 50,
    out_dir: str | Path = "sim_output",
    cfg: SimConfig | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if cfg is None:
        cfg = SimConfig()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    dfs: List[pd.DataFrame] = []

    for ep in range(n_episodes):
        ep_df = simulate_episode(ep, cfg, rng)
        dfs.append(ep_df)

    all_df = pd.concat(dfs, ignore_index=True)

    episode_summary = (
        all_df.groupby("episode_id")
        .agg(
            n_steps=("t", "count"),
            final_loc_error=("loc_error", "last"),
            avg_loc_error=("loc_error", "mean"),
            final_true_dist=("true_dist_to_goal", "last"),
            final_est_dist=("est_dist_to_goal", "last"),
            ever_act=("act_now", "max"),
            ever_success=("success_if_act_now", "max"),
            done_reason=("done_reason", "last"),
        )
        .reset_index()
    )

    all_df.to_csv(out_path / "time_series.csv", index=False)
    episode_summary.to_csv(out_path / "episode_summary.csv", index=False)

    # Save config for reproducibility
    pd.Series(asdict(cfg)).to_json(out_path / "config.json", indent=2)

    return all_df, episode_summary


# ============================================================
# Example main
# ============================================================


if __name__ == "__main__":
    cfg = SimConfig(
        dt=0.1,
        max_steps=350,
        n_landmarks=18,
        sensor_range=9.0,
        seed=7,
    )
    ts, summary = run_experiment(n_episodes=25, out_dir="sim_output", cfg=cfg)
    print("Saved:")
    print("  sim_output/time_series.csv")
    print("  sim_output/episode_summary.csv")
    print()
    print(summary.head())
