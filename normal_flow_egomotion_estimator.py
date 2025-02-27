import numpy as np
import torch
from scipy.linalg import logm, norm
from scipy.spatial.transform import Rotation as R
from sklearn.svm import LinearSVC
from utils import FlowData, Motion

class NormalFlowEgomotionEstimator:
    def __init__(self, use_cuda=False):
        """
        Initialize the class with input data.

        :param use_cuda: Wether to use CUDA or not.
        """
        # Fetch parameters.
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        # Parameters for fix-point iteration (FP).
        self.fix_pt_max_iter   = int(50)
        self.fix_pt_eta_vel    = float(1e-4)
        self.fix_pt_eta_omega  = float(1e-4)
        self.fix_pt_iter_break = int(0)

        # Constraints parameters.
        self.lamda = 1e-2  # Decreasing this gives more emphasis to co-point vectors.
        self.alpha = 1e+4  # Increasing this gives more emphasis to positive-depth values.

        # Patches matrix.
        self.p_matrix = torch.from_numpy(self.compute_patches(260, 346, 30)).int().to(self.device)

    def __call__(self, nflow: FlowData, vel_init: np.array, omega_init: np.array):
        """
        Sets the normal flow and initial omega,
        and then proceeds to call the egomotion method.

        :param nflow: Normal flow data.
        :param vel_init: Initial guess for velocity.
        :param omega_init: Initial guess for rotation.
        """
        # Fetch parameters.
        self.nflow         = nflow
        self.vel_init      = (torch.from_numpy(vel_init).double().to(self.device)
                              if vel_init is not None else None)
        self.omega_init    = torch.from_numpy(omega_init).double().to(self.device)

        # Fetch normal flow data to PyTorch tensors for optimization.
        self.x = torch.from_numpy(self.nflow.undist_x).double().to(self.device)
        self.y = torch.from_numpy(self.nflow.undist_y).double().to(self.device)
        self.u = torch.from_numpy(self.nflow.undist_u).double().to(self.device)
        self.v = torch.from_numpy(self.nflow.undist_v).double().to(self.device)

        # Segment the image field using regular patches.
        self.i = torch.from_numpy(self.nflow.dist_y).int().to(self.device)
        self.j = torch.from_numpy(self.nflow.dist_x).int().to(self.device)
        self.p = self.p_matrix[self.i, self.j]  # Pre-computed patch indices for each sample.

        # Compute unique patches.
        c, p_map, p_counts = torch.unique(self.p, return_inverse=True, return_counts=True)

        # Create a mask to keep only patches with enough input data points.
        mask = torch.isin(p_map, torch.nonzero(p_counts > 1))

        # Keep only valid data.
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.u = self.u[mask]
        self.v = self.v[mask]
        self.i = self.i[mask]
        self.j = self.j[mask]
        self.p = self.p[mask]

        # Recompute new indices.
        self.c, self.p_map = torch.unique(self.p, return_inverse=True)
        self.num_patches   = self.c.numel()

        # Compute magnitude and gradient direction.
        self.r = torch.sqrt(self.u ** 2 + self.v ** 2)
        self.n = torch.stack((self.u / self.r, self.v / self.r), dim=1)

        # Precompute flow velocity coefficients.
        self.vel_coeffs = torch.zeros((self.x.shape[0], 2, 3), dtype=torch.double)

        self.vel_coeffs[:, 0, 0] = -1.0
        self.vel_coeffs[:, 1, 0] = 0.0

        self.vel_coeffs[:, 0, 1] = 0.0
        self.vel_coeffs[:, 1, 1] = -1.0

        self.vel_coeffs[:, 0, 2] = self.x
        self.vel_coeffs[:, 1, 2] = self.y

        # Precompute flow omega coefficients.
        self.omega_coeffs = torch.zeros((self.x.shape[0], 2, 3), dtype=torch.double)

        self.omega_coeffs[:, 0, 0] = self.x * self.y
        self.omega_coeffs[:, 1, 0] = 1.0 + self.y ** 2

        self.omega_coeffs[:, 0, 1] = -(1.0 + self.x ** 2)
        self.omega_coeffs[:, 1, 1] = -(self.x * self.y)

        self.omega_coeffs[:, 0, 2] = self.y
        self.omega_coeffs[:, 1, 2] = -self.x

        # To estimate.
        self.vel_est    = self.vel_init
        self.omega_est  = self.omega_init
        self.motion_est = Motion(vel=self.vel_est, omega=self.omega_est)

        return self.estimate()

    def residual_velocity(self, params):
        """
        Compute the residual for the velocity optimization
        using the planar-depth patches, co-point vectors,
        and positive depth constraints.

        Needs to have a well precomputed omega estimation (stored in self.omega_est).

        :param params: Parameters to optimize [tx, ty].
        :return: Residual vector.
        """
        # Fetch parameters to optimize.
        vel_proj = params  # [tx, ty].

        # Recover the 3d normalized translation using the inverse stereographic method.
        d = 1 + vel_proj[0] ** 2 + vel_proj[1] ** 2
        vel_est = torch.stack([
            2 * vel_proj[0] / d,
            2 * vel_proj[1] / d,
            (vel_proj[0] ** 2 + vel_proj[1] ** 2 - 1) / d
        ], dim=0)

        # Get the translational and rotational flow part.
        ut = self.vel_coeffs @ vel_est
        uw = self.omega_coeffs @ self.omega_est

        # Compute the translational and rotational normal flow part.
        nt = torch.einsum('ij,ij->i', self.n, ut)
        nw = torch.einsum('ij,ij->i', self.n, uw)

        # Compute the weights for co-point vectors.
        cos_psi = (ut * self.n).sum(dim=1) / (ut.norm(dim=1) * self.n.norm(dim=1))
        w = 1 / (cos_psi ** 2 + self.lamda)

        # Compute the best inverse depth patched.
        inv_z_patch_est = (
                torch.zeros(self.num_patches, dtype=torch.double).scatter_add(
                    0, self.p_map,
                    w * (self.r - nw) * nt) /
                torch.zeros(self.num_patches, dtype=torch.double).scatter_add(
                    0, self.p_map,
                    w * (nt ** 2))
        )  # Already in index order [Zp0, Zp1, ..., ZpN].

        # Set each pixel with its corresponding depth patched.
        inv_z_est = inv_z_patch_est[self.p_map]

        # Compute the residual using the positive depth as relaxation term.
        f = w * (self.r - inv_z_est * nt - nw) + self.alpha * torch.relu(-(self.r - nw) * nt) ** 2

        return f

    def residual_omega(self, params):
        """
        Compute the residual for the omega optimization
        using the planar-depth patches, co-point vectors,
        and positive depth constraints.

        Needs to have a well precomputed velocity estimation (stored in self.vel_est).

        :param params: Parameters to optimize [wx, wy, wz].
        :return: Residual vector.
        """
        # Fetch parameters to optimize.
        omega_est = params  # [wx, wy, wz].

        # Get the translational and rotational flow part.
        ut = self.vel_coeffs @ self.vel_est
        uw = self.omega_coeffs @ omega_est

        # Compute the translational and rotational normal flow part.
        nt = torch.einsum('ij,ij->i', self.n, ut)
        nw = torch.einsum('ij,ij->i', self.n, uw)

        # Compute the weights for co-point vectors.
        cos_psi = (ut * self.n).sum(dim=1) / (ut.norm(dim=1) * self.n.norm(dim=1))
        w = 1 / (cos_psi ** 2 + self.lamda)

        # Compute the best inverse depth patched.
        inv_z_patch_est = (
                torch.zeros(self.num_patches, dtype=torch.double).scatter_add(
                    0, self.p_map,
                    w * (self.r - nw) * nt) /
                torch.zeros(self.num_patches, dtype=torch.double).scatter_add(
                    0, self.p_map,
                    w * (nt ** 2))
        )  # Already in index order [Zp0, Zp1, ..., ZpN].

        # Set each pixel with its corresponding depth patched.
        inv_z_est = inv_z_patch_est[self.p_map]

        # Compute the residual using the positive depth as relaxation term.
        f = w * (self.r - inv_z_est * nt - nw) + self.alpha * torch.relu(-(self.r - nw) * nt) ** 2

        return f

    def compute_velocity_prior(self):
        """
        Estimate the velocity prior with
        Linear Support Vector Classification (SVC)
        using the positive-depth constraint.

        Needs to have a well precomputed omega estimation (stored in self.omega_est).

        :return: Estimated scaled linear velocity prior (3DoF).
        """
        if self.vel_init is None:
            # Subtract rotational normal flow.
            uw = self.omega_coeffs @ self.omega_est
            nt = self.r - torch.einsum('ij,ij->i', self.n, uw)

            # Get the projected translation matrix A(x).
            na = (self.n[:, None, :] @ self.vel_coeffs)[:, 0, :]

            # Get weights and sign for the constraint.
            weights = torch.abs(nt)
            sign_w  = torch.sign(nt)

            # Estimate the translation direction using SVC.
            x_balanced = np.concatenate([na, -na], axis=0)
            y_balanced = np.concatenate([sign_w, -sign_w], axis=0)
            s_balanced = np.concatenate([weights, weights], axis=0)

            res = LinearSVC(fit_intercept=False, C=1).fit(x_balanced,
                                                          y_balanced,
                                                          sample_weight=s_balanced)

            self.vel_init = torch.tensor(res.coef_, dtype=torch.double).squeeze()
            self.vel_init = self.vel_init / self.vel_init.norm()

        self.vel_init_proj = torch.stack([
            self.vel_init[0] / (1 - self.vel_init[2] + torch.finfo(torch.double).eps),
            self.vel_init[1] / (1 - self.vel_init[2] + torch.finfo(torch.double).eps)
        ], dim=0)

        self.vel_est = self.vel_init  # Initial velocity estimation.

        return self.vel_est

    def estimate_velocity(self):
        """
        Estimate the linear velocity
        through non-linear optimization.

        :return: Estimated scaled linear velocity (3DoF).
        """
        # To find the minimum thus the apparent translation, we perform a
        # minimization over the 2D space of epipole positions.
        params = torch.stack([
            self.vel_est[0] / (1 - self.vel_est[2] + torch.finfo(torch.double).eps),
            self.vel_est[1] / (1 - self.vel_est[2] + torch.finfo(torch.double).eps)
        ], dim=0)  # Get the corresponding velocity first estimate.

        # Perform non-linear optimization.
        bounds = (self.vel_init_proj * 0.9, self.vel_init_proj * 1.1)
        params_est, _ = self.least_squares_torch(
            self.residual_velocity,
            params,
            lr=0.1,
            bounds=bounds
        )

        # Recover the 3d normalized translation using the inverse stereographic method.
        d = 1 + params_est[0] ** 2 + params_est[1] ** 2
        self.vel_est = torch.stack([
            2 * params_est[0] / d,
            2 * params_est[1] / d,
            (params_est[0] ** 2 + params_est[1] ** 2 - 1) / d
        ], dim=0)

        return self.vel_est

    def estimate_omega(self):
        """
        Estimate the omega
        through non-linear optimization.

        :return: Estimated omega (3DoF).
        """
        params = self.omega_est  # Get the corresponding omega first estimate.

        # Perform non-linear optimization.
        bounds = (self.omega_init * 0.9, self.omega_init * 1.1)
        params_est, _ = self.least_squares_torch(
            self.residual_omega,
            params,
            lr=0.1,
            bounds=bounds
        )

        self.omega_est = params_est

        return self.omega_est

    def estimate(self):
        """
        Estimate the self-motion from normal flow.

        :return: (Estimated motion (6DoF), iterations).
        """
        # Step 1: Compute the velocity prior.
        self.compute_velocity_prior()

        # Step 2: Perform minimization in a fixed-point iterative bilinear fashion.
        vel_old   = self.vel_est
        omega_old = self.omega_est
        for i in range(self.fix_pt_max_iter):
            # Refine the omega with the current estimated velocity.
            omega = self.estimate_omega()
            # Refine the velocity with the current estimated omega.
            vel = self.estimate_velocity()
            # Use L1-norm to check for convergence.
            if (max(abs(vel - vel_old)) < self.fix_pt_eta_vel and
                max(abs(omega - omega_old)) < self.fix_pt_eta_omega):
                self.fix_pt_iter_break = i + 1
                break
            vel_old   = vel
            omega_old = omega

        # Retrieve estimated motion.
        self.motion_est = Motion(
            vel=self.vel_est.detach().cpu().numpy(),
            omega=self.omega_est.detach().cpu().numpy()
        )

        return self.motion_est, self.fix_pt_iter_break

    @staticmethod
    def compute_patches(rows, cols, patch_size):
        """
        Perform image plane patche matrix.

        :param rows: number of row pixels.
        :param cols: number of col pixels.
        :param patch_size: size of the squared patch.
        :return: Patch matrix.
        """
        # Number of patches in each dimension
        num_patches_y = int(np.ceil(rows / patch_size))
        num_patches_x = int(np.ceil(cols / patch_size))

        # Number of patches
        K = num_patches_y * num_patches_x
        patches_matrix = np.zeros((rows, cols), dtype=int)

        # Build indices
        for idx in range(K):
            # Compute the patch indices
            ii = (idx // num_patches_x) * patch_size
            jj = (idx % num_patches_x) * patch_size
            # Fix the patch dimensions
            k = min(patch_size, rows - ii)
            w = min(patch_size, cols - jj)
            # Build patches indices matrix
            patches_matrix[ii:ii + k, jj:jj + w] = idx

        return patches_matrix
    
    @staticmethod
    def least_squares_torch(residual_func, initial_params, lr, max_iter=20, bounds=None, line_search_fn="strong_wolfe"):
        """
        Perform a least-squares optimization.

        :param residual_func: Function to optimize.
        :param initial_params: Parameters to optimize.
        :param lr: Learning rate.
        :param max_iter: Number of iterations.
        :param bounds: Lower and upper bounds.
        :param line_search_fn: Line Search.
        :return: Optimized parameters.
        """
        params = initial_params.clone().detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [params],
            lr=lr,
            line_search_fn=line_search_fn,
            max_iter=max_iter,
        )

        def closure():
            optimizer.zero_grad()  # Reset gradients to zero.

            # Compute residuals and loss (sum of squared residuals).
            residuals = residual_func(params)
            loss = torch.sum(residuals ** 2)
            loss.backward(retain_graph=True)  # Back-propagate the gradient.

            return loss

        # Optimization loop.
        optimizer.step(closure)  # Perform optimization step.
        lower_bounds, upper_bounds = bounds
        with torch.no_grad():
            params.clamp_(min=lower_bounds, max=upper_bounds)

        res = torch.sum(residual_func(params) ** 2).item()  # Final residual value.

        return params, res

    @staticmethod
    def print_motion(motion: Motion):
        """
        Prints the velocity (Vel) and rotational velocity (Omega) from the motion structure.

        :param motion: Motion.
        """
        vel   = motion.vel
        omega = motion.omega

        # Print velocity (Vel), normalized and formatted.
        norm_vel = np.linalg.norm(vel)
        print(f"\tVel=({vel[0] / norm_vel:.3e}, {vel[1] / norm_vel:.3e}, {vel[2] / norm_vel:.3e})m/frame")

        # Print rotational velocity (Omega) in degrees/frame, converting from radians to degrees.
        omega_deg = omega * 180 / np.pi
        print(f"\tOmega=({omega_deg[0]:.3e}, {omega_deg[1]:.3e}, {omega_deg[2]:.3e})deg/frame")

    def compute_relative_pose_error(self, motion_gt: Motion):
        """
        Computes the Relative Pose Error (RPE) between estimated and ground truth motion.

        :param motion_gt: Dictionary containing the ground truth motion ('Vel')
        :return: Relative Pose Error (RPE) in radians
        """
        # Extract velocities.
        vel_est = self.motion_est.vel.astype(np.float64)
        vel_gt  = motion_gt.vel.astype(np.float64)

        # Get the velocities norm.
        norm_est = np.linalg.norm(vel_est)
        norm_gt  = np.linalg.norm(vel_gt)

        # Compute the dot product and angle between the estimated and ground truth velocities.
        a = np.dot(vel_est, vel_gt) / (norm_est * norm_gt)

        # Calculate RPE using acos, with min to ensure the value is in the range [-1, 1].
        rpe = np.arccos(min(a, 1))

        return rpe

    def compute_relative_rotation_error(self, motion_gt: Motion):
        """
        Computes the Relative Rotation Error (RRE) between estimated and ground truth motion.

        :param motion_gt: Dictionary containing the ground truth motion ('Omega' in radians).
        :return: Relative Rotation Error (RRE).
        """
        # Extract omegas.
        omega_est = self.motion_est.omega.astype(np.float64)
        omega_gt  = motion_gt.omega.astype(np.float64)

        # Convert Omega (Euler angles) to rotation matrices.
        rot_est = R.from_euler('xyz', omega_est).as_matrix()
        rot_gt  = R.from_euler('xyz', omega_gt).as_matrix()

        # Compute the relative rotation matrix.
        relative_rot = rot_est.T @ rot_gt

        # Compute the matrix logarithm of the relative rotation.
        log_rot = logm(relative_rot)

        # Compute the norm of the matrix logarithm (Frobenius norm).
        rre = norm(log_rot)

        return rre