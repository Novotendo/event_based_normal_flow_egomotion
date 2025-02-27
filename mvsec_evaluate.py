import numpy as np
import os
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import Motion, FlowData
from normal_flow_egomotion_estimator import NormalFlowEgomotionEstimator
from scipy.interpolate import splprep, splev

def rms(x):
    return np.sqrt(np.mean(np.square(x)))

# Load MVSEC dataset files.
scene_path = 'data/mvsec/indoor_flying/indoor_flying3'
# scene_path = 'data/mvsec/outdoor_day/outdoor_day1'

scene_name    = os.path.basename(os.path.dirname(scene_path))
sequence_name = os.path.basename(scene_path)
print(f"Loading {sequence_name} from MVSEC dataset...")

odom       = np.load(f'{scene_path}/{sequence_name}_odom.npz')
timestamps = odom["timestamps"]
data_raw   = h5py.File(f'{scene_path}/{sequence_name}_data.hdf5')
imu        = data_raw['davis']['left']["imu"]
imu_t      = data_raw['davis']['left']["imu_ts"]

###############################################################################################################
#-------------------------------------------------TEST--------------------------------------------------------#
###############################################################################################################
all_timestamps = []
all_vel_preds_m, all_vel_preds_smooth_m, all_vel_gts_m = [], [], []
all_omega_preds_deg, all_omega_preds_smooth_deg, all_omega_gts_deg = [], [], []

# Load data preprocessed.
dist_events_xy    = np.load(f"{scene_path}/dataset_events_xy.npy")
events_t          = np.load(f"{scene_path}/dataset_events_t.npy")
undist_events_xy  = np.load(f"{scene_path}/undistorted_events_xy.npy")
undist_nflow_pred = np.load(f"{scene_path}/dataset_pred_flow_MVSEC.npy")
nflow_uncertainty = np.load(f"{scene_path}/dataset_angle_vars_flow_MVSEC.npy")

# Mask normal flow predictions based on uncertainty.
undist_nflow_pred[nflow_uncertainty > 0.3] = np.nan

# Instantiate the event-based normal flow egomotion estimator.
nflow_ego_estimator = NormalFlowEgomotionEstimator()

# Run test.
if sequence_name == "indoor_flying1":
    init_frame = 60
    last_frame = 1340
elif sequence_name == "indoor_flying2":
    init_frame = 150
    last_frame = 1500
elif sequence_name == "indoor_flying3":
    init_frame = 110
    last_frame = 1710
elif sequence_name == "outdoor_day1":
    init_frame = 1
    last_frame = 5100

for i in tqdm(range(init_frame, last_frame)):
    # Get timestamps between motion.
    t_0 = timestamps[i - 1]
    t_1 = timestamps[i]

    # Get the events between motion.
    events_t_0 = np.searchsorted(events_t, t_0)
    events_t_1 = np.searchsorted(events_t, t_1)
    events_idx = np.arange(events_t_0, events_t_1)

    # Get predicted normal flow event data.
    dist_x    = dist_events_xy[events_idx, 0]
    dist_y    = dist_events_xy[events_idx, 1]
    undist_x  = undist_events_xy[events_idx, 0]
    undist_y  = undist_events_xy[events_idx, 1]
    undist_un = undist_nflow_pred[events_idx, 0]
    undist_vn = undist_nflow_pred[events_idx, 1]

    # Create flow structure and mask it.
    mask = np.sqrt(undist_un ** 2 + undist_vn ** 2) > 1e-6
    nflow_est = FlowData(
        dist_x=dist_x[mask],
        dist_y=dist_y[mask],
        dist_u=None,
        dist_v=None,
        undist_x=undist_x[mask],
        undist_y=undist_y[mask],
        undist_u=undist_un[mask],
        undist_v=undist_vn[mask]
    )

    # Create ground-truth motion structure.
    motion_gt = Motion(
        vel=odom['lin_vel'][i],
        omega=odom['ang_vel'][i]
    )

    # Set initial values for the minimization.
    if (i - init_frame) % 4 == 0:
        # Set this to None to let the estimator compute the prior velocity estimation.
        vel_est   = None
        # Use "slow" rate IMU when available as initial omega estimation.
        imu_t_0   = np.searchsorted(imu_t, t_0)
        imu_t_1   = np.searchsorted(imu_t, t_1)
        imu_idx   = np.arange(imu_t_0, imu_t_1)
        omega_est = np.mean(imu[imu_idx, 3:], axis=0)
        # For outdoor_day sequence.
        # omega_est = np.mean(np.array([imu[imu_idx, 4], -imu[imu_idx, 3], imu[imu_idx, 5]]).T, axis=0)
    else:
        # In the meantime, use previous estimations.
        vel_est   = all_vel_preds_m[-1] / np.linalg.norm(all_vel_preds_m[-1])
        omega_est = all_omega_preds_deg[-1] * np.pi / 180  # rad/s

    # Estimate egomotion.
    motion_est, _ = nflow_ego_estimator(nflow_est, vel_est, omega_est)

    # Scale motion to m/s and deg/s.
    vel_pred_m     = motion_est.vel * np.linalg.norm(motion_gt.vel)  # Scaled with ground-truth.
    omega_pred_deg = motion_est.omega * 180 / np.pi

    # Apply a simple low-pass filter to the predicted motion.
    alpha_vel   = 0.4
    alpha_omega = 0.4
    if all_vel_preds_m:
        vel_pred_m = alpha_vel * all_vel_preds_m[-1] + (1 - alpha_vel) * vel_pred_m
    if all_omega_preds_deg:
        omega_pred_deg = alpha_omega * all_omega_preds_deg[-1] + (1 - alpha_omega) * omega_pred_deg

    all_vel_preds_m.append(vel_pred_m)
    all_vel_gts_m.append(motion_gt.vel)
    all_omega_preds_deg.append(omega_pred_deg)
    all_omega_gts_deg.append(motion_gt.omega * 180 / np.pi)
    all_timestamps.append(t_0 - timestamps[init_frame - 1])

    # Enforce smooth motion by B-spline interpolation.
    fit_samples = 20
    if len(all_vel_preds_m) > fit_samples:
        vel_tck, _   = splprep(np.array(all_vel_preds_m[-fit_samples:]).T,
                               u=np.array(all_timestamps[-fit_samples:]), s=30)
        omega_tck, _ = splprep(np.array(all_omega_preds_deg[-fit_samples:]).T,
                               u=np.array(all_timestamps[-fit_samples:]), s=30)

        vel_preds_smooth_m     = np.column_stack(splev(np.array(all_timestamps), vel_tck))
        omega_preds_smooth_deg = np.column_stack(splev(np.array(all_timestamps), omega_tck))

        all_vel_preds_smooth_m.append(vel_preds_smooth_m[-1])
        all_omega_preds_smooth_deg.append(omega_preds_smooth_deg[-1])
    else:
        all_vel_preds_smooth_m.append(all_vel_preds_m[-1])
        all_omega_preds_smooth_deg.append(all_omega_preds_deg[-1])

# Plot results.
t_init      = 0.0
t_last      = timestamps[last_frame] - timestamps[init_frame - 1]
range_vel   = 1.5
range_omega = 30.0

plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.title(f'Linear Velocity (m/s) Prediction vs Ground Truth', fontsize=20)
plt.plot(np.array(all_timestamps), np.array(all_vel_preds_smooth_m)[:, 0], label='PRED', color='red', linewidth=2)
plt.plot(np.array(all_timestamps), np.array(all_vel_gts_m)[:, 0], label='GT', color='black', linewidth=2)
plt.xlim(t_init, t_last)
plt.ylim(-range_vel, range_vel)
plt.ylabel('vx (m/s)', fontsize=12)
plt.legend()
plt.xticks(np.arange(t_init, t_last, 10.0))

plt.subplot(3, 1, 2)
plt.plot(np.array(all_timestamps), np.array(all_vel_preds_smooth_m)[:, 1], label='pred', color='red', linewidth=2)
plt.plot(np.array(all_timestamps), np.array(all_vel_gts_m)[:, 1], label='gt', color='black', linewidth=2)
plt.xlim(t_init, t_last)
plt.ylim(-range_vel, range_vel)
plt.ylabel('vy (m/s)', fontsize=12)
plt.xticks(np.arange(t_init, t_last, 10.0))

plt.subplot(3, 1, 3)
plt.plot(np.array(all_timestamps), np.array(all_vel_preds_smooth_m)[:, 2], label='pred', color='red', linewidth=2)
plt.plot(np.array(all_timestamps), np.array(all_vel_gts_m)[:, 2], label='gt', color='black', linewidth=2)
plt.xlim(t_init, t_last)
plt.ylim(-range_vel, range_vel)
plt.ylabel('vz (m/s)', fontsize=12)
plt.xlabel('Time (s)', fontsize=12)
plt.xticks(np.arange(t_init, t_last, 10.0))
plt.tight_layout()
plt.savefig(f'plot/{os.path.basename(scene_path)}/ego_xyz/{str(i - init_frame).zfill(6)}.png')
plt.close()

plt.figure(figsize=(10, 10))
plt.subplot(3, 1, 1)
plt.title(f'Angular Velocity (deg/s) Prediction vs Ground Truth', fontsize=20)
plt.plot(np.array(all_timestamps), np.array(all_omega_preds_smooth_deg)[:, 0], label='PRED', color='red', linewidth=2)
plt.plot(np.array(all_timestamps), np.array(all_omega_gts_deg)[:, 0], label='GT', color='black', linewidth=2)
plt.xlim(t_init, t_last)
plt.ylim(-range_omega, range_omega)
plt.ylabel('wx (deg/s)', fontsize=12)
plt.legend()
plt.xticks(np.arange(t_init, t_last, 10.0))

plt.subplot(3, 1, 2)
plt.plot(np.array(all_timestamps), np.array(all_omega_preds_smooth_deg)[:, 1], label='pred', color='red', linewidth=2)
plt.plot(np.array(all_timestamps), np.array(all_omega_gts_deg)[:, 1], label='gt', color='black', linewidth=2)
plt.xlim(t_init, t_last)
plt.ylim(-range_omega, range_omega)
plt.ylabel('wy (deg/s)', fontsize=12)
plt.xticks(np.arange(t_init, t_last, 10.0))

plt.subplot(3, 1, 3)
plt.plot(np.array(all_timestamps), np.array(all_omega_preds_smooth_deg)[:, 2], label='pred', color='red', linewidth=2)
plt.plot(np.array(all_timestamps), np.array(all_omega_gts_deg)[:, 2], label='gt', color='black', linewidth=2)
plt.xlim(t_init, t_last)
plt.ylim(-range_omega, range_omega)
plt.ylabel('wz (deg/s)', fontsize=12)
plt.xlabel('Time (s)', fontsize=12)
plt.xticks(np.arange(t_init, t_last, 10.0))
plt.tight_layout()
plt.savefig(f'plot/{os.path.basename(scene_path)}/ego_omega/{str(i - init_frame).zfill(6)}.png')
plt.close()

# Compute RMS errors.
rms_vel   = rms(np.array(all_vel_preds_smooth_m) - np.array(all_vel_gts_m))
rms_omega = rms(np.array(all_omega_preds_smooth_deg) - np.array(all_omega_gts_deg))
print("\033[93m" + (
    f"RMS Vel: {rms_vel:.3f} m/s | "
    f"RMS Omega: {rms_omega:.3f} deg/s"
) + "\033[0m")