import math
import numpy as np
import torch

class Analytic(object):
    """
    Example 4.2.
    """
    def __init__(self, device):
        
        self.dim = 2
        self.TT = 1
        self.X0_ub = 1
        self.ODE_solver = 'RK23'
        self.data_tol = 1e-08
        self.max_nodes = 5000
        self.tseq = np.linspace(0., self.TT, 20+1)[1:]
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
        X0 =  np.random.rand(self.dim, num_samples)
        X0 = (self.X0_ub - self.X0_lb) * X0 + self.X0_lb 
        if num_samples == 1:
            X0 = X0.flatten()
            
        return X0
    


    def _sqeuc(self, x):
        return torch.sum(x * x, dim=1, keepdim=True)
    
    def _prod(self, x, y):
        return torch.sum(x * y, dim=1, keepdim=True)

    def ham(self, tt, xx, pp):
        """
        The Hamiltonian.
        """
        out = -1/4 * self._sqeuc(pp) +  self._prod(pp, xx)

        return out
    
    def U_star(self, X_aug):
        '''Control as a function of the costate.'''
        Ax = X_aug[self.dim:2*self.dim]
        U = -1/2 * Ax
        return U
    
    def make_bc(self, X0_in):
        def bc(X_aug_0, X_aug_T):
            X0 = X_aug_0[:self.dim]
            XT = X_aug_T[:self.dim]
            AT = X_aug_T[self.dim:2*self.dim]
            vT = X_aug_T[2*self.dim:]

            # Derivative of the terminal cost with respect to X(T)
            dFdXT = 2*XT 

            return np.concatenate((X0 - X0_in, AT - dFdXT, vT))
        return bc
    
    def dynamics(self, t, X, U_fun):
        '''Evaluation of the dynamics at a single time instance for closed-loop
        ODE integration.'''
        U = U_fun([[t]], X.reshape((1,-1))).flatten()
        return  X + U
    
    def terminal_cost(self, X):

        return  np.sum(X.reshape((-1,1)) * X.reshape((-1,1)), axis=0, keepdims=True)
    
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
        dxdt =  x + U

        dAxdt = - Ax 
        
        L = self.running_cost(x, U)
        
        return np.vstack((dxdt, dAxdt, -L))


    def get_trace(self, grad, xx, batch_size, dim, grad_outputs_vec):
        """
        Computation of the second-order term in the HJB equation.
        """
        hess_stripes = torch.autograd.grad(outputs=grad, inputs=xx,
                                           grad_outputs=grad_outputs_vec,
                                           create_graph=True, retain_graph=True, only_inputs=True)[0]
        pre_laplacian = torch.stack([hess_stripes[i * batch_size: (i + 1) * batch_size, i]
                                     for i in range(0, dim)], dim=1)
        laplacian = torch.sum(pre_laplacian, dim=1)
        laplacian_sum_repeat = laplacian.repeat((1, dim))

        return laplacian_sum_repeat.T

    def psi_func(self, xx_inp):
        """
        The final-time cost function.
        """

        return self._sqeuc(xx_inp)
    
    
    def V_exact(self, x, t):
        return 2*self._sqeuc(x)/(1+ math.exp(2*(t-1)))