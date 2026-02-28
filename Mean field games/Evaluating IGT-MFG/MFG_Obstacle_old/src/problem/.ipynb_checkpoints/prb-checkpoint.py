import math
import numpy as np
import torch

class Obstacle(object):
    """
    Example 4.2.
    """
    def __init__(self, G_NN_list, Round, n, x0_initial, device, VV):
        
        self.dim = 2
        self.TT = 1.
        self.X0_ub = 1
        self.c = 3/2
        self.gamma_obst = 3
        self.gamma_cong = 3
        self.psi_scale = 3/2
        self.target = [[0.75, 0.]]
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

        # ---------- réplication de x0 ----------
        x_rep  = x0.repeat(timesteps + 1,1)                      # ((T+1)*N,d)
        x_test = torch.tensor(x_rep, dtype=torch.float32, device=self.device)

        # ---------- sorties de tous les générateurs ----------
        X_out_all = []
        for G in self.G_NN_list:
            with torch.no_grad():
                X = G(t_test, x_test).cpu()                                   # ((T+1)*N,d)
            X_out_all.append(X.view(timesteps + 1, num_samples, d))           # (T+1,N,d)
        X_out_all = torch.stack(X_out_all, dim=0)       # (K,T+1,N,d)

        # ---------- échantillonnage de n_vis trajectoires ----------
        gen_idx    = rng.integers(0, K, size=n_vis)            # (n_vis,)
        sample_idx = rng.integers(0, num_samples, size=n_vis)  # (n_vis,)

        # (n_vis, T+1, d)  →  transpose puis reshape
        pts = X_out_all[gen_idx, :, sample_idx, :].permute(1, 0, 2)           # (T+1,n_vis,d)
        x_batch = pts.reshape(-1, d)                                          # ((T+1)*n_vis, d)

        # idem pour t
        t_repeat = t_grid.unsqueeze(1).repeat(1, num_samples).reshape(-1, 1)        # ((T+1)*n_vis,1)

        return t_repeat.detach().requires_grad_(True), x_batch.to(self.device).detach().requires_grad_(True)
    
    def sample_x0(self, num_samples, noise_std=0.001):

        valid_samples = []
        max_attempts = num_samples * 10  # Avoid infinite loops, limit attempts
        attempts = 0

        while len(valid_samples) < num_samples and attempts < max_attempts:
            # Generate a random sample in [-1,1] x [-1,1]
            x = np.random.uniform(-1, 1)
            y = np.random.uniform(-0.5, 0.5)


            # Reject points inside the obstacle (out < 0 means inside)
            if self.FF_obstacle_func(x, y) <= 0:  # Accept only if outside the obstacle
                valid_samples.append([x, y])

            
            attempts += 1

        valid_samples = np.array(valid_samples)

        if len(valid_samples) < num_samples:
            print(f"Warning: Only {len(valid_samples)} samples generated out of {num_samples} requested.")

        return torch.tensor(valid_samples, dtype=torch.float32, device=self.device)
        

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
        return (1 / smoothness) * torch.log(1 + torch.exp(smoothness * x))

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
        """
        batch_size = xx_inp.size(0)
        xx = xx_inp[:, 0:2]
        dim = xx.size(1)
        assert dim == 2, f"Require dim=2 but got dim={dim} (BAD)"

        # Two diagonal obstacles
        # Rotation matrix
        theta = torch.tensor(np.pi / 0.5, device=self.device)
        rot_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                [torch.sin(theta), torch.cos(theta)]],
                               device=self.device).expand(batch_size, dim, dim)

        # Bottom/Left obstacle
        center1 = torch.tensor([-0., 0.3], dtype=torch.float, device=self.device)
        xxcent1 = xx - center1
        xxcent1 = xxcent1.unsqueeze(1).bmm(rot_mat).squeeze(1)
        covar_mat1 = torch.eye(dim, dtype=torch.float, device=self.device)
        covar_mat1[0:2, 0:2] = torch.tensor([[10, 0], [0, 1]], dtype=torch.float, device=self.device)
        covar_mat1 = covar_mat1.expand(batch_size, dim, dim)
        bb_vec1 = torch.tensor([0, 3], dtype=torch.float, device=self.device).expand_as(xx)
        xxcov1 = xxcent1.unsqueeze(1).bmm(covar_mat1)
        quad1 = torch.bmm(xxcov1, xxcent1.unsqueeze(2)).view(-1, 1)
        lin1 = torch.sum(xxcent1 * bb_vec1, dim=1, keepdim=True)
        out1 = (-1) * ((quad1 + lin1) + 1)
        out1 = scale * out1.view(-1, 1)
        out1 =  self.smooth_clamp_min(out1)

        # Top/Right obstacle
        center2 = torch.tensor([0., -0.3], dtype=torch.float, device=self.device)
        xxcent2 = xx - center2
        xxcent2 = xxcent2.unsqueeze(1).bmm(rot_mat).squeeze(1)
        covar_mat2 = torch.eye(dim, dtype=torch.float, device=self.device)
        covar_mat2[0:2, 0:2] = torch.tensor([[10, 0], [0, 1]], dtype=torch.float, device=self.device)
        covar_mat2 = covar_mat2.expand(batch_size, dim, dim)
        bb_vec2 = torch.tensor([0, -3], dtype=torch.float, device=self.device).expand_as(xx)
        xxcov2 = xxcent2.unsqueeze(1).bmm(covar_mat2)
        quad2 = torch.bmm(xxcov2, xxcent2.unsqueeze(2)).view(-1, 1)
        lin2 = torch.sum(xxcent2 * bb_vec2, dim=1, keepdim=True)
        out2 = (-1) * ((quad2 + lin2) + 1)
        out2 = scale * out2.view(-1, 1)
        out2 = self.smooth_clamp_min(out2)

        out = out1 + out2

        return out
    
    

    def compute_loss_gradients(self, xx_inp):
        """
        Compute gradients of the loss w.r.t. input.
        """
        # Ensure the input tensor requires gradient
        xx_inp = xx_inp.clone().detach().requires_grad_(True)

        # Compute the loss
        loss = self.F_obstacle_func_loss(xx_inp)

        # Compute gradients using autograd
        loss.backward(torch.ones_like(loss))

        # Extract gradients for x and y
        loss_gradient_x1 = xx_inp.grad[:, 0]  # Gradient w.r.t. x
        loss_gradient_x2 = xx_inp.grad[:, 1]  # Gradient w.r.t. y


        # Convert gradients to NumPy arrays and return them
        return np.array([loss_gradient_x1.view(-1).cpu().numpy(), loss_gradient_x2.view(-1).cpu().numpy()])

    
    def Conv_tor(self, t, x):
        
        cov_rho_m = []
        t_expanded = t.repeat_interleave(self.x0_initial.shape[0]).view(-1,1)
        x0_expanded = self.x0_initial.repeat(t.shape[0],1)
        
        if self.VV == 1:
            if self.Round == 0: 
                cov = self.rho(x[:, 0:2].reshape(t.shape[0],1,2) - x0_expanded[:,0:2].reshape(t.shape[0], self.x0_initial.shape[0], 2))
                cov = torch.mean(cov, dim=1)
                cov_rho_m.append(cov)
                
                if self.n == 0:
                    return cov_rho_m[self.n]
                else:
                    for i in range(1, self.n + 1):
                        New_dist = self.G_NN_list[i-1](t_expanded, x0_expanded)
                        new_cov = self.rho(x[:, 0:2].reshape(t.shape[0],1,2) - New_dist[:,0:2].reshape(t.shape[0], self.x0_initial.shape[0],2))
                        new_cov = torch.mean(new_cov, dim=1)
                        cov_rho_m.append((1 / (i + 1)) * new_cov + (i / (i + 1)) * cov_rho_m[i-1]) 
                        #cov_rho_m.append((1/2) * new_cov + (1/2) * cov_rho_m[i-1]) 
                    return cov_rho_m[self.n]
            else:
                New_dist = self.G_NN_list[0](t_expanded, x0_expanded)
                cov_0 = self.rho(x[:, 0:2].reshape(t.shape[0],1,2) - New_dist[:,0:2].reshape(t.shape[0], self.x0_initial.shape[0],2))
                cov_0 = torch.mean(cov_0, dim=1)
                cov_rho_m.append(cov_0)
                if self.n == 0: 
                    return cov_rho_m[self.n]
                else:
                    for i in range(1, self.n + 1):
                        New_dist = self.G_NN_list[i](t_expanded, x0_expanded)
                        new_cov = self.rho(x[:, 0:2].reshape(t.shape[0],1,2) - New_dist[:,0:2].reshape(t.shape[0], self.x0_initial.shape[0],2))
                        new_cov = torch.mean(new_cov, dim=1)
                        cov_rho_m.append((1 / (i + 1)) * new_cov + (i / (i + 1)) * cov_rho_m[i-1])
                        #cov_rho_m.append((1/2) * new_cov + (1/2) * cov_rho_m[i-1])
                    return cov_rho_m[self.n]
        else:
            New_dist = self.G_NN_list[-1](t_expanded, x0_expanded)
            cov_f = self.rho(x[:, 0:2].reshape(t.shape[0],1,2) - New_dist[:,0:2].reshape(t.shape[0], self.x0_initial.shape[0],2))
            cov_f = torch.mean(cov_f, dim=1)   
            return cov_f

        
    def compute_Conv_tor_gradients(self, t, x):
        
        # Ensure `x` requires gradients
        t = t.clone().detach().requires_grad_(True)
        x = x.clone().detach().requires_grad_(True)

        # Compute the output of Conv_tor
        output = self.Conv_tor(t, x)       

        output.backward(torch.ones_like(output))

        # Extract gradients with respect to x0 and x1
        grad_x0 = x.grad[:, 0]  # Gradient with respect to x0
        grad_x1 = x.grad[:, 1]  # Gradient with respect to x1

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
            dFdXT = np.zeros_like(AT)
            dFdXT[:2] = 2 * self.psi_scale * (XT[:2] - target_array)

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

        z = (X[:2]- np.array(self.target).squeeze())

        return self.psi_scale*np.sum(z * z, axis=0, keepdims=True)


    def running_cost(self,t, X, U):
        
        FF = self.F_obstacle_func_loss(torch.tensor(X.T, dtype=torch.float32, device=self.device)).cpu().detach().numpy()
        Conv = self.Conv_tor(torch.tensor(t, dtype=torch.float32, device=self.device), torch.tensor(X.T, dtype=torch.float32, device=self.device)).cpu().detach().numpy()
        
        return self.c * np.sum(U * U, axis=0, keepdims=True) +  self.gamma_obst * FF.T + self.gamma_cong * Conv.reshape((1,-1))

    
    
    def aug_dynamics(self, t, X_aug):
        
        '''Evaluation of the augmented dynamics at a vector of time instances for solution of the two-point BVP.'''
        
        U = self.U_star(X_aug)
       
        x = X_aug[:self.dim]

        
        dFF =  self.compute_loss_gradients(torch.tensor(x.T, dtype=torch.float32, device=self.device))

        dCov = self.compute_Conv_tor_gradients(torch.tensor(t, dtype=torch.float32, device=self.device), torch.tensor(x.T, dtype=torch.float32, device=self.device))

         
        Ax = X_aug[self.dim:2*self.dim]

        
        dxdt =  2 * self.c * U
        
        dAxdt = np.zeros_like(dxdt)
        
        
        dAxdt[0:2,:] = - (self.gamma_obst * dFF + self.gamma_cong * dCov)
        

        L = self.running_cost(t, x, U)

       
        fun = np.vstack((dxdt, dAxdt, -L))


        return fun
    

    
    def terminal(self, xx_inp):
        
        z = (xx_inp[:,0:2]-torch.tensor(self.target, device=self.device))
        
        return self.psi_scale * torch.sum(z*z, dim=1, keepdim=True)
    

    def psi_func(self, xx_inp):
        """
        The final-time cost function.
        """
        
        return self.terminal( xx_inp)
    

            
    
    
