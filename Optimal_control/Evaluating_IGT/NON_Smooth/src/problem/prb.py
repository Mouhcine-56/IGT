import math
import numpy as np
import torch

class Analytic(object):
    """
    Example 4.2.
    """
    def __init__(self, device):
        
        self.dim = 100
        self.TT = 1
        self.X0_ub = 1
        self.ODE_solver = 'RK23'
        self.data_tol = 1e-08
        self.max_nodes = 5000
        self.X0_lb = - self.X0_ub
        self.device = device


       

    def sample_x0(self, num_samples):
        """
        The initial distribution rho_0 of the agents.
        """
        X0 = torch.rand(num_samples, self.dim, device=self.device)
        X0 = (self.X0_ub - self.X0_lb) * X0 + self.X0_lb
       
        return X0
    
    def gen_x0(self, num_samples):
        """
        The initial distribution rho_0 of the agents.
        """
        X0 =  np.random.rand(num_samples, self.dim)
        X0 =  (self.X0_ub - self.X0_lb) * X0 + self.X0_lb
        if num_samples == 1:
            X0 = X0.flatten()
            
        return X0


    def _sqeuc(self, x):
        return torch.sum(x * x, dim=1, keepdim=True)

    def ham(self, tt, xx, pp):
        
        """
        Hamiltonian for L = 1 (PyTorch version).
        """

        # pp is tensor of shape (..., n)
        norm_p = torch.norm(pp, dim=-1)

        inside = (norm_p < 2.0)

        H_inside = - (norm_p**2) / 4.0
        H_outside = - norm_p + 1.0

        return torch.where(inside, H_inside, H_outside).view(-1,1)


    def U_star(self, X_aug):
        """Control as a function of the costate (column-wise)."""
        p = X_aug[self.dim:2*self.dim, :]                     # shape (dim, m)
        norm_p = np.linalg.norm(p, axis=0, keepdims=True)

        U = np.zeros_like(p)

        mask_in = norm_p < 2.0
        mask_out = norm_p > 2.0
        mask_eq = ~mask_in & ~mask_out

        U[:, mask_in[0]] = -0.5 * p[:, mask_in[0]]
        U[:, mask_out[0]] = - p[:, mask_out[0]] / norm_p[:, mask_out[0]]
        U[:, mask_eq[0]] = 0.0

        return U
    
    def make_bc(self, X0_in):
        def bc(X_aug_0, X_aug_T):
            X0 = X_aug_0[:self.dim]
            XT = X_aug_T[:self.dim]
            AT = X_aug_T[self.dim:2*self.dim]
            vT = X_aug_T[2*self.dim:]

            # Derivative of the terminal cost with respect to X(T)
            dFdXT = -2*XT 

            return np.concatenate((X0 - X0_in, AT - dFdXT, vT))
        return bc
    
    def dynamics(self, t, X, U_fun):
        '''Evaluation of the dynamics at a single time instance for closed-loop
        ODE integration.'''
        U = U_fun(float(t), X.reshape((-1, 1))).flatten()
        return U
    
    def terminal_cost(self, X):

        return  - np.sum(X * X, axis=0, keepdims=True)
    
    def running_cost(self, X, U):

        return  np.sum(U * U, axis=0, keepdims=True)

    
    def aug_dynamics(self, t, X_aug):
        '''Evaluation of the augmented dynamics at a vector of time instances
        for solution of the two-point BVP.'''

        U = self.U_star(X_aug)
        
        x = X_aug[:self.dim]

        # Costate
        Ax = X_aug[self.dim:2*self.dim]

        # Matrix-vector multiplication for all samples
        dxdt =  U

        dAxdt = np.zeros_like(dxdt)
        
        L = self.running_cost(x, U)
   
        return np.vstack((dxdt, dAxdt, -L))


    def psi_func(self, xx_inp):
        """
        The final-time cost function.
        """

        return - self._sqeuc(xx_inp)    
    
    def V_exact(self, x, t):
        """
        Exact value function V(t,x) for arbitrary horizon T = self.TT.
        Handles all regimes of L = T - t relative to the unit control bound.
        """
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        t = torch.as_tensor(t, dtype=torch.float32, device=self.device)

        if x.ndim == 1:
            x = x.unsqueeze(0)
        if t.ndim == 0:
            t = t.expand(x.shape[0]).unsqueeze(1)
        elif t.ndim == 1:
            t = t.unsqueeze(1)

        t = t.clamp(min=0.0, max=self.TT)
        L = (self.TT - t).clamp(min=0.0)

        norm_x = torch.norm(x, dim=-1, keepdim=True)
        V = torch.zeros_like(norm_x)

        eps = 1e-8
        mask_tT = torch.isclose(t, torch.full_like(t, self.TT), atol=eps)
        mask_mid = ~mask_tT

        # Final time: V(T,x) = -||x||^2
        V = torch.where(mask_tT, -norm_x**2, V)

        if not torch.any(mask_mid):
            return V

        mask_mid = mask_mid.squeeze(-1)
        L_mid = L[mask_mid]
        norm_mid = norm_x[mask_mid]
        V_mid = V[mask_mid]

        mask_L_lt1 = L_mid < (1.0 - eps)
        mask_L_eq1 = torch.isclose(L_mid, torch.ones_like(L_mid), atol=eps)
        mask_L_gt1 = L_mid > (1.0 + eps)

        # Case 0 < L < 1
        if torch.any(mask_L_lt1):
            L_lt1 = L_mid[mask_L_lt1]
            norm_lt1 = norm_mid[mask_L_lt1]

            thresh = (1.0 - L_lt1).clamp(min=eps)
            mask_inside = norm_lt1 <= thresh
            denom = thresh
            V_in = -norm_lt1**2 / denom
            V_out = -norm_lt1**2 - 2.0 * L_lt1 * norm_lt1 + L_lt1 * (1.0 - L_lt1)

            V_mid_lt1 = torch.where(mask_inside, V_in, V_out)
            V_mid[mask_L_lt1] = V_mid_lt1

        # Case L = 1
        if torch.any(mask_L_eq1):
            norm_eq1 = norm_mid[mask_L_eq1]
            mask_zero = norm_eq1 <= eps
            V_eq1 = torch.where(mask_zero, torch.zeros_like(norm_eq1),
                                -norm_eq1**2 - 2.0 * norm_eq1)
            V_mid[mask_L_eq1] = V_eq1

        # Case L > 1
        if torch.any(mask_L_gt1):
            L_gt1 = L_mid[mask_L_gt1]
            norm_gt1 = norm_mid[mask_L_gt1]
            mask_zero = norm_gt1 <= eps

            V_gt_nonzero = -norm_gt1**2 - 2.0 * L_gt1 * norm_gt1 + L_gt1 - L_gt1**2
            V_gt_zero = -L_gt1 * (L_gt1 - 1.0)

            V_gt = torch.where(mask_zero, V_gt_zero, V_gt_nonzero)
            V_mid[mask_L_gt1] = V_gt

        V[mask_mid] = V_mid
        return V


