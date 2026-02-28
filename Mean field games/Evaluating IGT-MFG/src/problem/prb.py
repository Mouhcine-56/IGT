import math
import numpy as np
import torch
import gc

from scipy.interpolate import interp1d

class Analytic(object):
    """
    Example 4.2.
    """
    def __init__(self, G_NN_list, Round, n, x0_initial, device, VV):

        self.dim = 50
        self.TT = 1
        self.X0_ub = 2
        self.ODE_solver = 'RK23'
        self.data_tol = 1e-07
        self.max_nodes = 5000
        self.tseq = np.linspace(0., self.TT, 20+1)[1:]
        self.X0_lb = - self.X0_ub
        self.device = device
        self.x0_initial = x0_initial
        self.mean_0 = torch.mean(self.x0_initial) * 0 + 1
        self.G_NN_list = G_NN_list
        self.Round = Round
        self.n = n
        self.sigma = np.sqrt(0.105)
        self.mu = 0.1
        self.VV = VV

        # Precomputed values for optimization
        self.N_particles = x0_initial.shape[0]

        # Mean interpolator
        self._mean_interp = None


    def _compute_mean_internal(self, t, x0_batch):
        """Internal method that accepts a specific batch of x0."""
        n_times = t.shape[0]
        N = x0_batch.shape[0]

        t_expanded = t.repeat_interleave(N).view(-1, 1)
        x0_expanded = x0_batch.repeat(n_times, 1)

        if self.VV == 1:
            X_t = self.G_NN_list[0](t_expanded, x0_expanded)
            mean_accum = X_t.view(n_times, N, self.dim).mean(dim=1)

            for i in range(1, self.n + 1):
                X_t = self.G_NN_list[i](t_expanded, x0_expanded)
                mean_new = X_t.view(n_times, N, self.dim).mean(dim=1)
                alpha = 1.0 / (i + 1)
                mean_accum = alpha * mean_new + (1.0 - alpha) * mean_accum
            return mean_accum
        else:
            X_t = self.G_NN_list[-1](t_expanded, x0_expanded)
            return X_t.view(n_times, N, self.dim).mean(dim=1)

    def precompute_mean(self, n_points=100, n_particles_precise=20000):
        """
        Precompute the mean field.
        If dim=1: Deterministic computation on x0_initial (cubic interpolation).
        If dim>1: Stochastic computation with large batch (linear interpolation).
        """
        if len(self.G_NN_list) == 0:
            return

        t_grid = np.linspace(0, self.TT, n_points)
        t_tensor = torch.tensor(t_grid, dtype=torch.float32, device=self.device).view(-1, 1)

        if self.dim == 1:
            # --- Deterministic case (dim=1) ---
            with torch.no_grad():
                mean_vals = self._compute_mean_raw(t_tensor).cpu().numpy()

            # Cubic interpolation for smoothness in 1D
            self._mean_interp = interp1d(t_grid, mean_vals, axis=0, kind='cubic', fill_value='extrapolate')

        else:
            # --- Stochastic case (dim > 1) ---
            print(f"Precomputing precise mean with {n_particles_precise} particles...")

            batch_size = 2000

            n_batches = max(1, n_particles_precise // batch_size)
            mean_sum = 0

            with torch.no_grad():
                for b in range(n_batches):
                    x0_batch = self.gen_x0(batch_size, Torch=True)
                    mean_batch = self._compute_mean_internal(t_tensor, x0_batch)
                    mean_sum += mean_batch
                    # Free memory between batches
                    del x0_batch, mean_batch
                    if b % 5 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                mean_vals = (mean_sum / n_batches).cpu().numpy()
            del mean_sum, t_tensor
            gc.collect()
            torch.cuda.empty_cache()

            if mean_vals.ndim == 1:
                mean_vals = mean_vals.reshape(-1, self.dim)

            # Linear interpolation for robustness in high dimension
            self._mean_interp = interp1d(t_grid, mean_vals, axis=0, kind='linear', fill_value='extrapolate')
            del mean_vals, t_grid  # Free after creating interpolator
            gc.collect()

    def clear_mean(self):
        """Clear the mean interpolator."""
        self._mean_interp = None

    def _compute_mean_raw(self, t):
        """Compute the raw mean via G_NN networks."""
        n_times = t.shape[0]
        t_expanded = t.repeat_interleave(self.N_particles).view(-1, 1)
        x0_expanded = self.x0_initial.repeat(n_times, 1)

        if self.VV == 1:
            X_t = self.G_NN_list[0](t_expanded, x0_expanded)
            mean_accum = X_t.view(n_times, self.N_particles, self.dim).mean(dim=1)

            for i in range(1, self.n + 1):
                X_t = self.G_NN_list[i](t_expanded, x0_expanded)
                mean_new = X_t.view(n_times, self.N_particles, self.dim).mean(dim=1)
                alpha = 1.0 / (i + 1)
                mean_accum = alpha * mean_new + (1.0 - alpha) * mean_accum

            return mean_accum
        else:
            X_t = self.G_NN_list[-1](t_expanded, x0_expanded)
            return X_t.view(n_times, self.N_particles, self.dim).mean(dim=1)

    def update_mean(self, t, x):
        """Return the mean (interpolated if available)."""
        # Use the interpolator if available
        if self._mean_interp is not None:
            if isinstance(t, torch.Tensor):
                t_np = t.detach().cpu().numpy().flatten()
            else:
                t_np = np.atleast_1d(t).flatten()

            mean_val = self._mean_interp(t_np)
            if mean_val.ndim == 1:
                mean_val = mean_val.reshape(-1, self.dim)
            return torch.tensor(mean_val, dtype=torch.float32, device=self.device)

        # If no G_NN, return a constant
        if len(self.G_NN_list) == 0:
            n_times = t.shape[0]
            return torch.full((n_times, self.dim), self.mu, dtype=torch.float32, device=self.device)

        # Otherwise compute raw mean
        with torch.no_grad():
            return self._compute_mean_raw(t)

    def sample_x0(self, num_samples, Torch=True):
        X0 = torch.rand(num_samples, self.dim, device=self.device)
        X0 = (self.X0_ub - self.X0_lb) * X0 + self.X0_lb
        if Torch:
            return X0
        else:
            return X0.cpu().detach().numpy()

    def gen_x0(self, num_samples, Torch=False):
        samples = np.random.normal(loc=self.mu, scale=self.sigma, size=(num_samples, self.dim))
        if Torch:
            return torch.tensor(samples, dtype=torch.float32, device=self.device)
        else:
            return samples

    def _sqeuc(self, x):
        return torch.sum(x * x, dim=1, keepdim=True)

    def _prod(self, x, y):
        return torch.sum(x * y, dim=1, keepdim=True)

    def F(self, t, x):
        up_mean = self.update_mean(t, x)
        return 0.5 * self._sqeuc((x - up_mean))

    def d_F(self, t, x):
        up_mean = self.update_mean(t, x)
        return x - up_mean

    def ham(self, tt, xx, pp):
        out = -0.5 * self._sqeuc(pp) + self.F(tt, xx)
        return out

    def dynamics(self, t, X, U_fun):
        U = U_fun([[t]], X.reshape((1, -1))).flatten()
        return U

    def dynamics_torch(self, t, x, V_NN):
        U = -V_NN.get_grad(t, x)
        return U

    def U_star(self, X_aug):
        Ax = X_aug[self.dim:2*self.dim]
        U = -Ax
        return U

    def make_bc(self, X0_in):
        def bc(X_aug_0, X_aug_T):
            X0 = X_aug_0[:self.dim]
            AT = X_aug_T[self.dim:2*self.dim]
            vT = X_aug_T[2*self.dim:]
            dFdXT = 0
            return np.concatenate((X0 - X0_in, AT - dFdXT, vT))
        return bc

    def terminal_cost(self, X):
        return 0

    def running_cost(self, t, X, U):
        t_tensor = torch.tensor(np.atleast_1d(t), dtype=torch.float32, device=self.device).view(-1, 1)
        x_tensor = torch.tensor(X.T, dtype=torch.float32, device=self.device)
        FF = self.F(t_tensor, x_tensor).cpu().detach().numpy()
        return 0.5 * np.sum(U * U, axis=0, keepdims=True) + FF.T

    def aug_dynamics(self, t, X_aug):
        U = self.U_star(X_aug)
        x = X_aug[:self.dim]

        t_tensor = torch.tensor(np.atleast_1d(t), dtype=torch.float32, device=self.device).view(-1, 1)
        x_tensor = torch.tensor(x.T, dtype=torch.float32, device=self.device)

        dFF = self.d_F(t_tensor, x_tensor).cpu().detach().numpy().T

        Ax = X_aug[self.dim:2*self.dim]
        dxdt = U
        dAxdt = -dFF
        L = self.running_cost(t, x, U)

        return np.vstack((dxdt, dAxdt, -L))
