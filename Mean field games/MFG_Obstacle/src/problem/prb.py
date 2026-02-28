import math
import numpy as np
import torch

class Obstacle(object):
    """
    Example 4.2.
    """
    def __init__(self, G_NN_list, Round, n, x0_initial, device, VV):
        
        self.dim = 10
        self.TT = 1.
        self.X0_ub = 1
        self.c = 3/2
        self.gamma_obst = 3
        self.gamma_cong = 3
        self.psi_scale = 3/2
        self.target = [[0.75, 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
        self.ODE_solver = 'RK23'
        self.data_tol = 1e-04
        self.max_nodes = 5000
        self.tseq = np.linspace(0., self.TT, 20+1)[1:]
        self.X0_lb = - self.X0_ub
        self.device = device
        self.x0_initial = x0_initial
        self.G_NN_list = G_NN_list
        self.Round = Round
        self.n = n
        self.VV = VV
        self.sigma_rho = 0.5
        
        # Initialize cache variables
        self.cached_trajectories = []
        self.use_precomputed = False

    def clear_cache(self):
        """Clear cached trajectories and G_NN_list to free memory."""
        if hasattr(self, 'cached_trajectories'):
            del self.cached_trajectories
            self.cached_trajectories = []
        if hasattr(self, 'G_NN_list'):
            del self.G_NN_list
            self.G_NN_list = []
        self.use_precomputed = False
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def sample_mbar_points(self, num_samples):
        
        x0 = self.gen_x0(num_samples, Torch=True)

        rng = np.random.default_rng() 
        K   = len(self.G_NN_list)
        d   = x0.shape[1]
        
        if K <5:
            n_vis = K
        else:
            n_vis = 5
        
        timesteps = 20

        # ---------- grille t ----------
        t_grid = torch.linspace(0., 1., timesteps + 1, device=self.device)         # (T+1,)
        t_test = t_grid.repeat_interleave(num_samples).view(-1, 1)            # ((T+1)*N,1)

        # ---------- replicate x0 ----------
        x_rep  = x0.repeat(timesteps + 1,1)                      # ((T+1)*N,d)
        x_test = torch.tensor(x_rep, dtype=torch.float32, device=self.device)

        # ---------- sorties de tous les générateurs ----------
        X_out_all = []
        for G in self.G_NN_list:
            with torch.no_grad():
                X = G(t_test, x_test).cpu()                                   # ((T+1)*N,d)
            X_out_all.append(X.view(timesteps + 1, num_samples, d))           # (T+1,N,d)
        X_out_all = torch.stack(X_out_all, dim=0)       # (K,T+1,N,d)

        # ---------- sample n_vis trajectories ----------
        gen_idx    = rng.integers(0, K, size=n_vis)            # (n_vis,)
        sample_idx = rng.integers(0, num_samples, size=n_vis)  # (n_vis,)

        # (n_vis, T+1, d)  →  transpose puis reshape
        pts = X_out_all[gen_idx, :, sample_idx, :].permute(1, 0, 2)           # (T+1,n_vis,d)
        x_batch = pts.reshape(-1, d)                                          # ((T+1)*n_vis, d)

        # idem pour t
        t_repeat = t_grid.unsqueeze(1).repeat(1, num_samples).reshape(-1, 1)        # ((T+1)*n_vis,1)

        return t_repeat.detach().requires_grad_(True), x_batch.to(self.device).detach().requires_grad_(True)
    
    def sample_x0(self, num_samples, noise_std=0.01):
        """
        Sample points uniformly in the domain for HJB training.
        First 2 dims: uniform in [-1, 1] x [-1, 1]
        Remaining dims: Gaussian N(0, noise_std^2)
        
        Args:
            num_samples: Number of samples to generate
            noise_std: Std for higher dimensions (default 0.1)
            
        Returns:
            Tensor of shape (num_samples, self.dim)
        """
        # First 2 dimensions: uniform in [-1, 1]
        xy = np.random.uniform(-1, 1, size=(num_samples, 2))
        
        if self.dim > 2:
            # Remaining dimensions: Gaussian
            rest = noise_std * np.random.randn(num_samples, self.dim - 2)
            samples = np.hstack([xy, rest])
        else:
            samples = xy
        
        return torch.tensor(samples, dtype=torch.float32, device=self.device)
        
    def gen_x0(self, num_samples, Torch=False):

        mu = np.array([[-0.75, 0.] + [0] * (self.dim - 2)], dtype=np.float32)
        samples = np.sqrt(0.01) * np.random.randn(num_samples, self.dim) + mu

        if Torch:
            return torch.tensor(samples, dtype=torch.float32, device=self.device)
        else:
            return samples
        
        
    def rho(self, x):
        
        # Ensure x is a 2D tensor with shape (N, 2)
        if x.shape[-1] != 2:
            raise ValueError("Input x must have shape (N, 2) for 2D density.")

        # Compute the squared norm (x1^2 + x2^2) / (2 * sigma^2)
        squared_norm =  torch.sum(x**2, dim=-1) / (2 * (self.sigma_rho**2))

        # Compute the 2D Gaussian density
        normalization = 1 / (2 * math.pi * (self.sigma_rho**2))
        r = normalization * torch.exp(-squared_norm)

        return r


    def _sqeuc(self, x):
        return torch.sum(x * x, dim=1, keepdim=True)
    
    def _prod(self, x, y):
        return torch.sum(x * y, dim=1, keepdim=True)
    
    def circular_obstacle_torch(self, x, y, center, radius):
        """
        Defines a circular obstacle in PyTorch.
        """
        dist_squared = (x - center[0])**2 + (y - center[1])**2
        return radius**2 - dist_squared  # Differentiable expression

    def smooth_max_torch(self, f1, f2, smoothness=50.0):
        """
        Smooth max in PyTorch (differentiable).
        """
        return (1 / smoothness) * torch.log(torch.exp(smoothness * f1) + torch.exp(smoothness * f2))
    
    def smooth_clamp_min(self, x, smoothness=50.0):
        """
        Smooth approximation of torch.clamp_min(x, 0).

        Parameters:
        - x: Input tensor.
        - smoothness: Controls the sharpness of the transition (higher is sharper).
        """
        # Numerically stable softplus: log(1 + exp(x)) = x + log(1 + exp(-x)) for large x
        return torch.where(
            smoothness * x > 20,
            x,  # For large x, softplus(x) ≈ x
            (1 / smoothness) * torch.log(1 + torch.exp(smoothness * x))
        )

    def FF_obstacle_func(self, x, y):
        """
        Two-diagonal obstacles confined to the interval [-1, 1] x [-1, 1].
        """
        # Rotation matrix for diagonal obstacles (rotated by 36 degrees)
        theta = np.pi / 0.5
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)

        # Bottom/Left obstacle (adjusted center and scaling for [-1, 1] x [-1, 1])
        center1 = np.array([-0., 0.3], dtype=float)  # Center shifted within [-1, 1]
        vec1 = np.array([x, y], dtype=float) - center1
        vec1 = np.dot(vec1, rot_mat)  # Apply rotation
        mat1 = np.array([[10, 0], [0, 1]], dtype=float)  # Adjust scaling for [-1, 1]
        bb1 = np.array([0, 3], dtype=float)  # Adjust slope/linear term
        quad1 = np.dot(vec1, np.dot(mat1, vec1))
        lin1 = np.dot(vec1, bb1)
        out1 = np.clip((-1) * (quad1 + lin1 + 1), a_min=-0.1, a_max=None)

        # Top/Right obstacle (adjusted center and scaling for [-1, 1] x [-1, 1])
        center2 = np.array([0., -0.3], dtype=float)  # Center shifted within [-1, 1]
        vec2 = np.array([x, y], dtype=float) - center2
        vec2 = np.dot(vec2, rot_mat)  # Apply rotation
        mat2 = np.array([[10, 0], [0, 1]], dtype=float)  # Adjust scaling for [-1, 1]
        bb2 = np.array([0, -3], dtype=float)  # Adjust slope/linear term
        quad2 = np.dot(vec2, np.dot(mat2, vec2))
        lin2 = np.dot(vec2, bb2)
        out2 = np.clip((-1) * (quad2 + lin2 + 1), a_min=-0.1, a_max=None)

        # Combine the two obstacles
        out = out1 + out2

        return out

    def F_obstacle_func_loss(self, xx_inp, scale=1):
        """
        Calculate interaction term. Calculates F(x), where F is the forcing term in the HJB equation.
        Uses cached constant tensors for speed.
        """
        batch_size = xx_inp.size(0)
        xx = xx_inp[:, 0:2]
        dim = xx.size(1)
        assert dim == 2, f"Require dim=2 but got dim={dim} (BAD)"

        # Initialize cached tensors on first call
        if not hasattr(self, '_obs_rot_mat'):
            theta = torch.tensor(np.pi / 0.5, device=self.device)
            self._obs_rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                    [torch.sin(theta), torch.cos(theta)]],
                                   device=self.device)
            self._obs_center1 = torch.tensor([-0., 0.3], dtype=torch.float, device=self.device)
            self._obs_center2 = torch.tensor([0., -0.3], dtype=torch.float, device=self.device)
            self._obs_covar_mat = torch.tensor([[10., 0.], [0., 1.]], dtype=torch.float, device=self.device)
            self._obs_bb_vec1 = torch.tensor([0., 3.], dtype=torch.float, device=self.device)
            self._obs_bb_vec2 = torch.tensor([0., -3.], dtype=torch.float, device=self.device)

        # Expand rotation matrix for batch
        rot_mat = self._obs_rot_mat.unsqueeze(0).expand(batch_size, dim, dim)

        # Bottom/Left obstacle
        xxcent1 = xx - self._obs_center1
        xxcent1 = xxcent1.unsqueeze(1).bmm(rot_mat).squeeze(1)
        covar_mat1 = self._obs_covar_mat.unsqueeze(0).expand(batch_size, dim, dim)
        xxcov1 = xxcent1.unsqueeze(1).bmm(covar_mat1)
        quad1 = torch.bmm(xxcov1, xxcent1.unsqueeze(2)).view(-1, 1)
        lin1 = torch.sum(xxcent1 * self._obs_bb_vec1, dim=1, keepdim=True)
        out1 = (-1) * ((quad1 + lin1) + 1)
        out1 = scale * out1.view(-1, 1)
        out1 = self.smooth_clamp_min(out1)

        # Top/Right obstacle
        xxcent2 = xx - self._obs_center2
        xxcent2 = xxcent2.unsqueeze(1).bmm(rot_mat).squeeze(1)
        xxcov2 = xxcent2.unsqueeze(1).bmm(covar_mat1)
        quad2 = torch.bmm(xxcov2, xxcent2.unsqueeze(2)).view(-1, 1)
        lin2 = torch.sum(xxcent2 * self._obs_bb_vec2, dim=1, keepdim=True)
        out2 = (-1) * ((quad2 + lin2) + 1)
        out2 = scale * out2.view(-1, 1)
        out2 = self.smooth_clamp_min(out2)

        out = out1 + out2

        return out
    
    

    def compute_loss_gradients(self, xx_inp):
        """
        Compute gradients of the obstacle loss w.r.t. first 2 input dimensions.
        Optimized version using torch.autograd.grad instead of backward().
        """
        # Ensure the input tensor requires gradient
        xx_inp = xx_inp.detach().requires_grad_(True)

        # Compute the loss
        loss = self.F_obstacle_func_loss(xx_inp)

        # Compute gradients using autograd.grad (faster than backward)
        grads = torch.autograd.grad(
            outputs=loss,
            inputs=xx_inp,
            grad_outputs=torch.ones_like(loss),
            create_graph=False,
            retain_graph=False
        )[0]

        # Extract gradients for x and y
        loss_gradient_x1 = grads[:, 0]  # Gradient w.r.t. x
        loss_gradient_x2 = grads[:, 1]  # Gradient w.r.t. y

        # Convert gradients to NumPy arrays and return them
        return np.array([loss_gradient_x1.view(-1).cpu().numpy(), 
                        loss_gradient_x2.view(-1).cpu().numpy()])

    
    def precompute_congestion(self, time_res=51):
        """
        Precompute generator trajectories on a time grid for fast interpolation.
        
        This caches Y_i(t_k) = G_NN(t_k, x0_i) for all initial points x0_i 
        and time steps t_k. When evaluating Conv_tor at arbitrary t, we 
        interpolate these cached trajectories.
        
        Mathematical correctness:
        (rho * m)(x) = int rho(x - y) dm(y) ≈ (1/N) sum_i rho(x - y_i(t))
        
        We cache y_i(t_k) and interpolate to get y_i(t).
        """
        print("Precomputing generator trajectories...")
        
        self.time_res = time_res
        self.t_grid = torch.linspace(0, self.TT, time_res, device=self.device)  # (T,)
        
        N_x0 = self.x0_initial.shape[0]
        d = self.dim
        K = len(self.G_NN_list)
        
        if K == 0:
            self.use_precomputed = False
            print("No generators to precompute.")
            return
        
        # For each generator, store trajectories: shape (K, T, N_x0, d)
        # But we only need the weighted average trajectories based on VV and n
        
        # Expand time and x0 for batch evaluation
        # t_expanded: (T * N_x0, 1), x0_expanded: (T * N_x0, d)
        t_expanded = self.t_grid.repeat_interleave(N_x0).view(-1, 1)
        x0_expanded = self.x0_initial.repeat(time_res, 1)
        
        # Compute trajectories for all generators
        self.cached_trajectories = []  # List of (T, N_x0, 2) tensors
        
        with torch.no_grad():
            for G in self.G_NN_list:
                traj = G(t_expanded, x0_expanded)  # (T*N_x0, d)
                traj = traj[:, :2].reshape(time_res, N_x0, 2)  # (T, N_x0, 2)
                self.cached_trajectories.append(traj)
        
        self.use_precomputed = True
        print(f"Precomputed {K} generator trajectories on {time_res} time points.")

    def _interpolate_trajectories(self, t_query, requires_grad=False):
        """
        Interpolate cached trajectories at query times.
        
        Args:
            t_query: (M,) or (M, 1) tensor of query times
            requires_grad: if True, allow gradient computation through interpolation
            
        Returns:
            List of tensors, each of shape (M, N_x0, 2) - interpolated positions
        """
        if t_query.dim() == 2:
            t_query = t_query.squeeze(-1)  # (M,)
        
        M = t_query.shape[0]
        
        # Normalize t to [0, time_res-1] for interpolation indices
        t_normalized = t_query / self.TT * (self.time_res - 1)  # (M,)
        t_normalized = t_normalized.clamp(0, self.time_res - 1)
        
        # Get floor and ceil indices
        t_floor = t_normalized.floor().long()  # (M,)
        t_ceil = (t_floor + 1).clamp(max=self.time_res - 1)  # (M,)
        
        # Interpolation weight
        alpha = (t_normalized - t_floor.float()).view(M, 1, 1)  # (M, 1, 1)
        
        interpolated = []
        
        context = torch.no_grad() if not requires_grad else torch.enable_grad()
        with context:
            for traj in self.cached_trajectories:
                # traj: (T, N_x0, 2)
                traj_floor = traj[t_floor]  # (M, N_x0, 2)
                traj_ceil = traj[t_ceil]    # (M, N_x0, 2)
                
                # Linear interpolation
                traj_interp = (1 - alpha) * traj_floor + alpha * traj_ceil  # (M, N_x0, 2)
                interpolated.append(traj_interp)
        
        return interpolated

    def Conv_tor(self, t, x):
        """
        Compute convolution (rho * m)(x) at time t.
        
        Using Monte Carlo approximation:
        (rho * m)(x) ≈ (1/N) sum_i rho(x - y_i(t))
        
        where y_i(t) = G_NN(t, x0_i) are samples from the generator.
        
        Args:
            t: (Mt, 1) or (1, 1) time tensor. If (1, 1), broadcasts to all x points.
            x: (Nx, dim) spatial tensor
        """
        Nx = x.shape[0]  # Number of spatial points
        Mt = t.shape[0]  # Number of time points
        
        # Handle broadcasting: if t has 1 row but x has many, expand t
        if Mt == 1 and Nx > 1:
            t = t.expand(Nx, -1)
            Mt = Nx
        
        if getattr(self, 'use_precomputed', False) and len(self.cached_trajectories) > 0:
            # Use precomputed and interpolated trajectories
            N_x0 = self.x0_initial.shape[0]
            
            # Get interpolated trajectories for all generators at query times
            # Each element: (Mt, N_x0, 2)
            interp_trajs = self._interpolate_trajectories(t.view(-1))
            
            # x: (Mt, dim) -> (Mt, 1, 2) for broadcasting
            x_eval = x[:, :2].view(Mt, 1, 2)
            
            if self.VV == 1:
                # Compute weighted average of convolutions
                # cov_k = (1/N) sum_i rho(x - y_i^k(t))
                # m_bar_n = (1/(n+1)) sum_{k=0}^n cov_k (with recursive weighting)
                
                # First generator
                y_0 = interp_trajs[0]  # (Mt, N_x0, 2)
                cov_0 = self.rho(x_eval - y_0)  # (Mt, N_x0)
                cov_0 = torch.mean(cov_0, dim=1)  # (Mt,)
                
                if self.n == 0:
                    return cov_0
                
                cov_prev = cov_0
                for i in range(1, min(self.n + 1, len(interp_trajs))):
                    y_i = interp_trajs[i]  # (Mt, N_x0, 2)
                    new_cov = self.rho(x_eval - y_i)  # (Mt, N_x0)
                    new_cov = torch.mean(new_cov, dim=1)  # (Mt,)
                    cov_prev = (1 / (i + 1)) * new_cov + (i / (i + 1)) * cov_prev
                
                return cov_prev
            else:
                # VV == 2: Use only the last generator
                y_last = interp_trajs[-1]  # (Mt, N_x0, 2)
                cov_f = self.rho(x_eval - y_last)  # (Mt, N_x0)
                cov_f = torch.mean(cov_f, dim=1)  # (Mt,)
                return cov_f

        # Original computation (no precomputation)
        cov_rho_m = []
        t_expanded = t.repeat_interleave(self.x0_initial.shape[0]).view(-1,1)
        x0_expanded = self.x0_initial.repeat(Mt, 1)
        
        if self.VV == 1:
            
            New_dist = self.G_NN_list[0](t_expanded, x0_expanded)
            cov_0 = self.rho(x[:, 0:2].reshape(Mt,1,2) - New_dist[:,0:2].reshape(Mt, self.x0_initial.shape[0],2))
            cov_0 = torch.mean(cov_0, dim=1)
            cov_rho_m.append(cov_0)
            if self.n == 0: 
                return cov_rho_m[self.n]
            else:
                for i in range(1, self.n + 1):
                    New_dist = self.G_NN_list[i](t_expanded, x0_expanded)
                    new_cov = self.rho(x[:, 0:2].reshape(Mt,1,2) - New_dist[:,0:2].reshape(Mt, self.x0_initial.shape[0],2))
                    new_cov = torch.mean(new_cov, dim=1)
                    cov_rho_m.append((1 / (i + 1)) * new_cov + (i / (i + 1)) * cov_rho_m[i-1])
                return cov_rho_m[self.n]
        else:
            New_dist = self.G_NN_list[-1](t_expanded, x0_expanded)
            cov_f = self.rho(x[:, 0:2].reshape(Mt,1,2) - New_dist[:,0:2].reshape(Mt, self.x0_initial.shape[0],2))
            cov_f = torch.mean(cov_f, dim=1)   
            return cov_f

        
    def compute_Conv_tor_gradients(self, t, x):
        """
        Compute gradients of Conv_tor w.r.t. first 2 spatial dimensions.
        Uses analytical gradient of rho(x-y) = -rho(x-y) * (x-y) / sigma^2
        
        d/dx [ (1/N) sum_i rho(x - y_i) ] = (1/N) sum_i d_rho(x - y_i) / dx
                                          = (1/N) sum_i -rho(x-y_i) * (x-y_i) / sigma^2
        
        Args:
            t: (M, 1) or (1, 1) time tensor. If (1, 1), broadcasts to all x points.
            x: (Nx, dim) spatial tensor
        """
        Nx = x.shape[0]  # Number of spatial points
        Mt = t.shape[0]  # Number of time points
        sigma2 = self.sigma_rho ** 2
        
        # Handle broadcasting: if t has 1 row but x has many, expand t
        if Mt == 1 and Nx > 1:
            t = t.expand(Nx, -1)
            Mt = Nx
        
        if getattr(self, 'use_precomputed', False) and len(self.cached_trajectories) > 0:
            # Use precomputed trajectories - much faster
            N_x0 = self.x0_initial.shape[0]
            
            # Get interpolated trajectories
            interp_trajs = self._interpolate_trajectories(t.view(-1), requires_grad=False)
            
            # x: (Mt, dim) -> x_2d: (Mt, 1, 2)
            x_2d = x[:, :2].view(Mt, 1, 2)
            
            if self.VV == 1:
                # Weighted average over generators
                # First generator
                y_0 = interp_trajs[0]  # (Mt, N_x0, 2)
                diff_0 = x_2d - y_0  # (Mt, N_x0, 2)
                rho_0 = self.rho(diff_0)  # (Mt, N_x0)
                
                # Gradient: -rho * diff / sigma^2
                grad_0 = -rho_0.unsqueeze(-1) * diff_0 / sigma2  # (Mt, N_x0, 2)
                grad_0 = torch.mean(grad_0, dim=1)  # (Mt, 2)
                
                if self.n == 0:
                    return grad_0.T.cpu().detach().numpy()  # (2, Mt)
                
                grad_prev = grad_0
                for i in range(1, min(self.n + 1, len(interp_trajs))):
                    y_i = interp_trajs[i]  # (Mt, N_x0, 2)
                    diff_i = x_2d - y_i
                    rho_i = self.rho(diff_i)  # (Mt, N_x0)
                    grad_i = -rho_i.unsqueeze(-1) * diff_i / sigma2
                    grad_i = torch.mean(grad_i, dim=1)  # (Mt, 2)
                    grad_prev = (1 / (i + 1)) * grad_i + (i / (i + 1)) * grad_prev
                
                return grad_prev.T.cpu().detach().numpy()  # (2, Mt)
            else:
                # VV == 2: Use only the last generator
                y_last = interp_trajs[-1]  # (Mt, N_x0, 2)
                diff_last = x_2d - y_last
                rho_last = self.rho(diff_last)
                grad_last = -rho_last.unsqueeze(-1) * diff_last / sigma2
                grad_last = torch.mean(grad_last, dim=1)  # (Mt, 2)
                return grad_last.T.cpu().detach().numpy()  # (2, Mt)
        
        # Fallback: use autograd (slower)
        t_for_grad = t.clone().detach().requires_grad_(False)
        x_for_grad = x.clone().detach().requires_grad_(True)

        output = self.Conv_tor(t_for_grad, x_for_grad)       
        output.backward(torch.ones_like(output))

        grad_x0 = x_for_grad.grad[:, 0]
        grad_x1 = x_for_grad.grad[:, 1]

        return np.array([grad_x0.view(-1).cpu().detach().numpy(), grad_x1.view(-1).cpu().detach().numpy()])

                    

    def ham(self, tt, xx, pp):
        
        """ The Hamiltonian."""
        
        out = -self.c * self._sqeuc(pp)  + self.gamma_obst*self.F_obstacle_func_loss(xx) + self.gamma_cong * self.Conv_tor(tt, xx).view(-1,1)

        return out
    
    def U_star(self, X_aug):
        
        '''Control as a function of the costate.'''
        Ax = X_aug[self.dim:2*self.dim]
        U =  -Ax
        return U
    
    def make_bc(self, X0_in):
        def bc(X_aug_0, X_aug_T):
            # Extract components
            X0 = X_aug_0[:self.dim]
            XT = X_aug_T[:self.dim]
            AT = X_aug_T[self.dim:2*self.dim]
            vT = X_aug_T[2*self.dim:]

            # Debugging: Print shapes
            #print(f"X0 shape: {X0.shape}, XT shape: {XT.shape}, AT shape: {AT.shape}, vT shape: {vT.shape}")

            # Assertions to validate shapes
            assert X0.shape == (self.dim,), f"X0 shape mismatch: {X0.shape} vs {(self.dim,)}"
            assert XT.shape == (self.dim,), f"XT shape mismatch: {XT.shape} vs {(self.dim,)}"
            assert AT.shape == (self.dim,), f"AT shape mismatch: {AT.shape} vs {(self.dim,)}"
            assert vT.shape == (1,), f"vT shape mismatch: {vT.shape} vs (1,)"

            # Derivative of the terminal cost with respect to the final state
            target_array = np.array(self.target).squeeze()
            #assert target_array.shape == (self.dim,), f"Target shape mismatch: {target_array.shape} vs {(self.dim,)}"
            #dFdXT = np.zeros_like(AT)
            dFdXT = 2 * self.psi_scale * (XT - target_array)

            # Compute boundary condition residuals
            residuals = np.concatenate((X0 - X0_in, AT - dFdXT, vT))
            #print("Boundary condition residuals:", residuals)

            return residuals
        return bc
    
    def dynamics_torch(self, t, x, V_NN):
        
        '''Evaluation of the dynamics at a single time instance for closed-loop ODE integration.'''
        
        U =  -V_NN.get_grad(t, x)

        return 2 * self.c * U
    
    def dynamics(self, t, X, U_fun):
        
        '''Evaluation of the dynamics at a single time instance for closed-loop ODE integration.'''
        
        U = U_fun([[t]], X.reshape((1,-1))).flatten()

        return  2 * self.c * U
    
    
    def terminal_cost(self, X):
        """Terminal cost for BVP solver (numpy). Uses all dimensions."""
        target_array = np.array(self.target).squeeze()
        z = X - target_array
        return self.psi_scale * np.sum(z * z, axis=0, keepdims=True)


    def running_cost(self,t, X, U):
        
        FF = self.F_obstacle_func_loss(torch.tensor(X.T, dtype=torch.float32, device=self.device)).cpu().detach().numpy()
        Conv = self.Conv_tor(torch.tensor(t, dtype=torch.float32, device=self.device), torch.tensor(X.T, dtype=torch.float32, device=self.device)).cpu().detach().numpy()
        
        return self.c * np.sum(U * U, axis=0, keepdims=True) +  self.gamma_obst * FF.T + self.gamma_cong * Conv.reshape((1,-1))

    
    
    def aug_dynamics(self, t, X_aug):
        
        '''Evaluation of the augmented dynamics at a vector of time instances for solution of the two-point BVP.'''
        
        U = self.U_star(X_aug)
       
        x = X_aug[:self.dim]
        
        # Convert to tensors once
        N = x.shape[1] if x.ndim > 1 else 1
        
        # Handle both scalar t and array t
        if np.isscalar(t):
            t_tensor = torch.tensor([[t]], dtype=torch.float32, device=self.device)
        else:
            t_tensor = torch.tensor(t.reshape(-1, 1), dtype=torch.float32, device=self.device)
        
        x_tensor = torch.tensor(x.T if x.ndim > 1 else x.reshape(1, -1), 
                                dtype=torch.float32, device=self.device)
        
        dFF = self.compute_loss_gradients(x_tensor)
        dCov = self.compute_Conv_tor_gradients(t_tensor, x_tensor)

         
        Ax = X_aug[self.dim:2*self.dim]

        
        dxdt =  2 * self.c * U
        
        dAxdt = np.zeros_like(dxdt)
        
        
        dAxdt[0:2,:] = - (self.gamma_obst * dFF + self.gamma_cong * dCov)
        

        L = self.running_cost_fast(t_tensor, x_tensor, U)

       
        fun = np.vstack((dxdt, dAxdt, -L))


        return fun
    
    def running_cost_fast(self, t_tensor, x_tensor, U):
        """Fast running cost computation with pre-converted tensors."""
        with torch.no_grad():
            FF = self.F_obstacle_func_loss(x_tensor).cpu().numpy()
            Conv = self.Conv_tor(t_tensor, x_tensor).cpu().numpy()
        
        return self.c * np.sum(U * U, axis=0, keepdims=True) + self.gamma_obst * FF.T + self.gamma_cong * Conv.reshape((1,-1))

    

    
    def terminal(self, xx_inp):
        """Terminal cost for neural network training (torch). Uses all dimensions."""
        target_tensor = torch.tensor(self.target, dtype=torch.float32, device=self.device)
        z = xx_inp - target_tensor
        return self.psi_scale * torch.sum(z * z, dim=1, keepdim=True)
    

    def psi_func(self, xx_inp):
        """
        The final-time cost function.
        """
        return self.terminal(xx_inp)
    

            
    
    