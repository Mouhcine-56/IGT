import torch
import time
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
import torch.optim as optim
import copy
from scipy.stats import ks_2samp
import torch.nn as nn
import warnings
import ot
from geomloss import SamplesLoss

#================================ DGM_HJB =========================================#

def Solve_HJB(obs, V_NN, num_epoch, t, lr, num_samples, device):
    """
    Train the value network V_NN to minimize the HJB residual using unsupervised learning.

    Args:
        obs          : Obstacle problem object (defines Hamiltonian, samplers, etc.)
        V_NN        : Neural network model for V(t, x)
        num_epoch   : Number of training epochs
        t           : Time grid (torch.Tensor of shape [num_samples, 1])
        lr          : Learning rate
        num_samples : Number of spatial samples x
        device      : Torch device (cpu or cuda)

    Returns:
        V_NN        : Trained value network
    """

    V_NN.train()
    optimizer = optim.Adam(V_NN.parameters(), lr)

    x_rand = obs.sample_x0(num_samples).requires_grad_(True)
    T = obs.TT * torch.ones(num_samples, 1, device=device)

    old_loss = float('inf')
    loss_history = []

    for epoch in range(num_epoch + 1):
        t = t.requires_grad_(True)

        # Forward pass
        V_nn = V_NN(t, x_rand)

        # ∂V/∂t
        V_nn_t = torch.autograd.grad(
            outputs=V_nn, inputs=t,
            grad_outputs=torch.ones_like(V_nn),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # ∇V
        V_nn_x = torch.autograd.grad(
            outputs=V_nn, inputs=x_rand,
            grad_outputs=torch.ones_like(V_nn),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # HJB residual loss
        Loss = torch.mean((V_nn_t + obs.ham(t, x_rand, V_nn_x)) ** 2)

        # Backward pass and optimization
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        # Logging and resampling every 1000 steps
        if epoch % 1000 == 0:
            loss_history.append(old_loss)
            new_loss = Loss.item()
            print(f"Iteration {epoch:5d}: Loss = {new_loss:.4e}")

            if new_loss < min(loss_history):
                old_loss = new_loss
            else:
                # Resample x_rand if no improvement
                x_rand = obs.sample_x0(num_samples).requires_grad_(True)

    print()
    return V_NN

#================================  BVP + HJB =========================================#

def Approximate_v(obs, V_NN, data, num_epoch, t, lr, num_samples, Round, device):
    """
    Train the value network V_NN using:
    - HJB residual loss
    - Supervised loss on V(t,x)
    - Supervised loss on ∇V(t,x)

    Args:
        obs          : Obstacle problem instance
        V_NN        : Value function network
        data        : Dictionary with keys 't', 'X', 'V', 'A'
        num_epoch   : Number of training iterations
        t           : Time grid (torch tensor)
        lr          : Learning rate
        num_samples : Number of samples for HJB loss
        Round       : Round index (for logging only)
        device      : Torch device (CPU or CUDA)

    Returns:
        V_NN        : Trained value network
    """
    V_NN.train()
    optimizer = optim.Adam(V_NN.parameters(), lr)

    x_rand = obs.sample_x0(num_samples).requires_grad_(True)

    old_loss = float('inf')
    loss_history = []

    for epoch in range(num_epoch + 1):
        t = t.requires_grad_(True)

        # Forward pass: V(t, x)
        V_nn = V_NN(t, x_rand)

        # ∂V/∂t
        V_nn_t = torch.autograd.grad(
            outputs=V_nn, inputs=t,
            grad_outputs=torch.ones_like(V_nn),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # ∇V
        V_nn_x = torch.autograd.grad(
            outputs=V_nn, inputs=x_rand,
            grad_outputs=torch.ones_like(V_nn),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Loss components
        Loss_hjb = torch.mean((V_nn_t + obs.ham(t, x_rand, V_nn_x)) ** 2)
        Loss_v    = torch.mean((V_NN(data['t'], data['X']) - data['V']) ** 2)
        Loss_v_x  = torch.mean((V_NN.get_grad(data['t'], data['X']) - data['A']) ** 2)

        # Total loss (can be weighted if needed)
        Loss_total = Loss_hjb + Loss_v + Loss_v_x

        # Backward and optimize
        optimizer.zero_grad()
        Loss_total.backward()
        optimizer.step()

        # Logging every 1000 iterations
        if epoch % 1000 == 0:
            loss_history.append(old_loss)
            new_loss = Loss_total.item()
            print(f"Iteration {epoch:5d}: "
                  f"Loss_V = {Loss_v.item():.4e}, "
                  f"Loss_V_x = {Loss_v_x.item():.4e}, "
                  f"Loss_HJB = {Loss_hjb.item():.4e}, "
                  f"Loss_total = {Loss_total.item():.4e}")

            # Resample x_rand if no improvement
            if new_loss >= min(loss_history):
                x_rand = obs.sample_x0(num_samples).requires_grad_(True)
            old_loss = new_loss

    print()
    return V_NN


#================================ solve BVP ===========================================#

def generate_data(obs, V_NN, num_samples, t, lr, num_samples_hjb, device,  num_epoch=1000):
    
    def eval_u(t, x):
        u =  -V_NN.get_grad(t, x).detach().cpu().numpy()
        return u
    
    def bvp_guess(t, x):
        V_NN.eval()
        V = V_NN(torch.tensor(t, dtype=torch.float32, device=device), 
                 torch.tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy() 
        V_x = V_NN.get_grad(t, x).detach().cpu().numpy() 
        return V, V_x

    print('Generating data...')

    dim = obs.dim

    X_OUT = np.empty((0, dim))
    A_OUT = np.empty((0, dim))
    V_OUT = np.empty((0, 1))
    t_OUT = np.empty((0, 1))

    Ns_sol = 0  # Initialize the number of successful solutions
    failure_count = 0
    
    start_time = time.time()
    x0_int = obs.gen_x0(num_samples, Torch=False)

    # ----------------------------------------------------------------------

    while Ns_sol < num_samples:
        
        print('Solving TPBVP #', Ns_sol+1, '...', end='\r')
        
        max_failures = num_samples - Ns_sol

        X0 = x0_int[Ns_sol,:]
        bc = obs.make_bc(X0)



        # Integrates the closed-loop system (NN controller)

        SOL = solve_ivp(obs.dynamics, [0., obs.TT], X0,
                        method=obs.ODE_solver,
                        args=(eval_u,),
                        rtol=1e-08)

        V_guess, A_guess = bvp_guess(SOL.t.reshape(1,-1).T, SOL.y.T)
        
        try:
            # Solves the two-point boundary value problem

            X_aug_guess = np.vstack((SOL.y, A_guess.T, V_guess.T))
            SOL = solve_bvp(obs.aug_dynamics, bc, SOL.t, X_aug_guess,
                            verbose=0,
                            tol=obs.data_tol,
                            max_nodes=obs.max_nodes)
            # Save only successful solutions
            if SOL.success:
                V = SOL.y[-1:] + obs.terminal_cost(SOL.y[:dim, -1])
                t_OUT = np.vstack((t_OUT, SOL.x.reshape(1, -1).T))
                X_OUT = np.vstack((X_OUT, SOL.y[:dim].T))
                A_OUT = np.vstack((A_OUT, SOL.y[dim:2*dim].T))
                V_OUT = np.vstack((V_OUT, V.T))
                Ns_sol += 1
                #failure_count = 0  # Reset failure count on success
            else:
                print(f"Solver failed ({failure_count + 1}/{max_failures}): {SOL.message}")
                failure_count += 1
                if Ns_sol < 25: #45
                        V_NN = Solve_HJB1(obs, V_NN, num_epoch, t, lr, num_samples_hjb, device)
                if failure_count >= max_failures:
                    print("Maximum retries reached. Exiting.")
                    break
                # Generate a new initial condition for retry
                X0 = x0_int[Ns_sol+failure_count,:]


        except Warning as e:
            print(f"Warning encountered: {str(e)}. Skipping...")
            failure_count += 1
            if failure_count >= max_failures:
                print("Maximum warnings reached. Exiting.")
                break


    print('Generated', X_OUT.shape[0], 'data from', Ns_sol,
        'BVP solutions in %.1f' % (time.time() - start_time), 'sec \n')

    data = {'t': torch.tensor(t_OUT, dtype=torch.float32, device=device), 
            'X': torch.tensor(X_OUT, dtype=torch.float32, device=device), 
            'A': torch.tensor(A_OUT, dtype=torch.float32, device=device), 
            'V': torch.tensor(V_OUT, dtype=torch.float32, device=device)}

    return data, Ns_sol



   
#==========================   Simulate points   ==========================================#   


import numpy as np
import time
from scipy.integrate import solve_ivp

def sim_points(obs, V_NN, num_samples, N, t0, tf, device):
    """
    Simulate closed-loop trajectories using the trained V_NN policy.

    Args:
        obs         : Obstacle problem instance
        V_NN        : Trained value network
        num_samples : Number of initial states to simulate
        N           : Number of time steps
        t0, tf      : Initial and final simulation times
        device      : torch device (CPU or CUDA)

    Returns:
        t_tr   : Flattened training time vector [(N+1)*num_samples, 1]
        x_tr   : Repeated initial states for training input [(N+1)*num_samples, dim]
        t_OUT  : Time grid for each trajectory [num_samples, N+1]
        X_OUT  : Flattened trajectory values [(N+1)*num_samples, dim]
        x_out  : Last full trajectory concatenation (can be removed if unused)
    """

    def eval_u(t, x):
        """Closed-loop control: u = -∇V"""
        return -V_NN.get_grad(t, x).detach().cpu().numpy()

    X_OUT = np.empty((0, obs.dim))
    t_OUT = np.empty((0, 1))

    # Initial conditions sampled from generator
    data = obs.gen_x0(num_samples, Torch=False)
    x_out = []  # Will store full x_out trajectory stack if needed

    Ns_sol = 0
    start_time = time.time()

    print('Generating data via closed-loop simulation...')

    while Ns_sol < num_samples:
        print(f'Solving IVP #{Ns_sol + 1}...', end='\r')

        X0 = data[Ns_sol, :]

        # Integrate dynamics under NN controller
        SOL = solve_ivp(
            fun=obs.dynamics,
            t_span=[t0, tf],
            y0=X0,
            method='RK23',
            t_eval=np.linspace(t0, tf, N + 1),
            args=(eval_u,),
            rtol=1e-8
        )

        # Store simulation outputs
        t_OUT = np.vstack((t_OUT, SOL.t.reshape(-1, 1)))
        X_OUT = np.vstack((X_OUT, SOL.y.T))
        x_out.append(SOL.y.T)  # Store full trajectory if needed

        Ns_sol += 1

    # Reshape time and state trajectories for training
    t_train = t_OUT.reshape(num_samples, N + 1)                     # [num_samples, N+1]
    t_tr = t_train.T.flatten().reshape(-1, 1)                       # [(N+1)*num_samples, 1]
    x_tr = np.tile(data[:num_samples, :], (N + 1, 1))              # [(N+1)*num_samples, dim]

    X_OUT = X_OUT.reshape(num_samples, N + 1, obs.dim)
    X_OUT = X_OUT.transpose(1, 0, 2).reshape((N + 1) * num_samples, obs.dim)

    print(f'Generated {X_OUT.shape[0]} points from {Ns_sol} IVP solutions '
          f'in {time.time() - start_time:.1f} sec.\n')

    return t_tr, x_tr, t_OUT, X_OUT, x_out



#================================  Train Generator =========================================#

def Train_Gen(obs, G_NN, V_NN, t_tr, x_tr, X_OUT, num_epoch, t, lr, num_samples, Round, device):
    """
    Train the generator network G_NN to match simulated trajectory data (X_OUT)
    and satisfy the ODE dynamics constraint via residual loss.

    Args:
        obs         : Obstacle problem instance
        G_NN        : Generator neural network
        V_NN        : Trained value network
        t_tr        : Time grid for training (flattened)
        x_tr        : Initial samples for training
        X_OUT       : Target trajectory output
        num_epoch   : Number of training iterations
        t           : Time tensor for ODE loss (HJB residual)
        lr          : Learning rate
        num_samples : Number of samples for ODE matching
        Round       : Training round (for logging only)
        device      : Torch device (CPU or CUDA)

    Returns:
        G_NN_original : Copy of G_NN before training
        G_NN          : Updated generator network
    """
    G_NN.eval()
    G_NN_original = copy.deepcopy(G_NN)

    # Resample x_rand for ODE loss
    x_rand = obs.gen_x0(num_samples, Torch=True).requires_grad_(True)
    t = t.requires_grad_(True)

    # Prepare training tensors
    t_train = torch.tensor(t_tr, dtype=torch.float32, device=device).requires_grad_(True)
    X_train = torch.tensor(x_tr, dtype=torch.float32, device=device).requires_grad_(True)
    X_OUT   = torch.tensor(X_OUT, dtype=torch.float32, device=device).requires_grad_(True)

    G_NN.train()
    optimizer = optim.Adam(G_NN.parameters(), lr)

    best_loss = float('inf')

    for epoch in range(num_epoch + 1):
        # Compute generator time derivative ∂G/∂t
        G_nn_t = G_NN.grad_t(t, x_rand)

        # Residual dynamics loss (should satisfy dx/dt = f(x, u))
        Loss_ode = torch.mean((G_nn_t - obs.dynamics_torch(t, G_NN(t, x_rand), V_NN)) ** 2)

        # Supervised trajectory loss
        Loss_G = torch.mean((G_NN(t_train, X_train) - X_OUT) ** 2)

        # Weighted total loss (dynamics weight = 0.01 for dim=2)
        Loss_total = Loss_G + 0.01 * Loss_ode

        optimizer.zero_grad()
        Loss_total.backward()
        optimizer.step()

        # Logging every 1000 steps
        if epoch % 1000 == 0:
            new_loss = Loss_total.item()
            print(f"Iteration {epoch:5d} | "
                  f"Loss_G = {Loss_G.item():.4e} | "
                  f"Loss_ODE = {Loss_ode.item():.4e} | "
                  f"Loss_total = {new_loss:.4e}")

            if new_loss < best_loss:
                best_loss = new_loss
            else:
                # Resample if loss stagnates
                x_rand = obs.gen_x0(num_samples, Torch=True).requires_grad_(True)

    print()
    return G_NN_original, G_NN


#================================== IGT process ==============================#

def train_v_nn(an, V_NN, num_epochs, num_epoch_hjb, t, lr_v, num_samples_hjb, num_samples_bvp, round_num, device, VV):
    """
    Trains the value function network V_NN by:
    1. Solving the HJB residual (unsupervised)
    2. Generating TPBVP data
    3. Fitting V_NN using supervised TPBVP loss

    Args:
        an               : Analytic object
        V_NN             : Value network to train
        num_epochs       : Epochs for BVP training
        num_epoch_hjb    : Epochs for HJB training
        t                : Time grid (torch.Tensor)
        lr_v             : Learning rate
        num_samples_hjb  : Number of samples for HJB loss
        num_samples_bvp  : Number of samples for TPBVP generation
        round_num        : Current round (for logging)
        device           : torch.device
        VV               : Mode indicator (1 or 2 for V1/V2)

    Returns:
        V_NN             : Trained value network
    """

    print(f"\nTraining V{VV} via HJB...\n")

    # Step 1: Initialization 
    V_NN = Solve_HJB(an, V_NN, num_epoch_hjb, t, lr_v, num_samples_hjb, device)

    # Step 2: Generating
    attempts = 0
    Ns_sol = 0

    while Ns_sol < 25 and attempts < 2:
        N_sol_bvp = 0

        while N_sol_bvp < 1:
            data, Ns_sol = generate_data(an, V_NN, num_samples_bvp, t, lr_v, num_samples_hjb, device)
            N_sol_bvp = Ns_sol

            if N_sol_bvp == 0:
                print(f"Re-solving V{VV} via HJB due to insufficient BVP data...")
                V_NN = Solve_HJB1(an, V_NN, num_epoch_hjb, t, lr_v, num_samples_hjb, device)

        # Step 3: Training 
        print(f"\nTraining V{VV} via BVP Approximation...\n")
        V_NN = Approximate_v(an, V_NN, data, num_epochs, t, lr_v, num_samples_hjb, round_num, device)
        attempts += 1

    return V_NN

    
    
#==================================Compute J and V ==============================#


def Comp_J(obs, t, x, V_NN):
    
    t_expanded = t.repeat_interleave(x.shape[0]).view(-1,1)
    x0_expanded = x.repeat(t.shape[0],1)
    
    X_n = obs.G_NN_list[-1](t_expanded, x0_expanded)
     
    cov_rho_m = obs.rho((X_n[:, 0:2].reshape(t.shape[0], x.shape[0], 2)).unsqueeze(2) -  (X_n[:,0:2].reshape(t.shape[0], x.shape[0], 2)).unsqueeze(1))
    cov_rho_m = torch.mean(cov_rho_m, dim=2)
    
    f_obs = []
    for ti in range(t.shape[0]):
        f_obs_i = obs.F_obstacle_func_loss(X_n[:,0:2].reshape(t.shape[0], x.shape[0], 2)[ti,:,:])
        f_obs.append(f_obs_i)
        
    f_obs = torch.stack(f_obs, dim=0).squeeze(-1)


    u = -V_NN.get_grad(t_expanded, X_n).reshape(t.shape[0], x.shape[0], obs.dim)
    

    l = obs.c * torch.sum(u * u, dim=2) +  obs.gamma_obst * f_obs +  obs.gamma_cong * cov_rho_m
    
    g = obs.psi_func(X_n[:,0:2].reshape(t.shape[0], x.shape[0], 2)[-1,:,:]) 
    
    J = torch.mean(1/(t.shape[0]-1) * torch.sum(l[0:t.shape[0]-1, :], dim=0) + g.squeeze())
    
    return J.item()
    
def Comp_J(obs, t, x, V_NN):
    """
    Compute the population cost functional J for a given generator and value network.

    Args:
        obs    : Obstacle problem instance (contains G_NN_list and cost components)
        t      : Time grid tensor of shape [T, 1]
        x      : Initial samples tensor of shape [N, dim]
        V_NN   : Value function network

    Returns:
        J (float) : Mean cost over the simulated population
    """
    # Expand (t, x) to shape [T*N, 1] and [T*N, dim] for G_NN input
    t_expanded = t.repeat_interleave(x.shape[0], dim=0)         # [T*N, 1]
    x0_expanded = x.repeat(t.shape[0], 1)                        # [T*N, dim]

    # Simulate trajectories using latest generator
    X_n = obs.G_NN_list[-1](t_expanded, x0_expanded)              # [T*N, dim]
    X_n_reshaped = X_n[:, :2].reshape(t.shape[0], x.shape[0], 2) # [T, N, 2]

    # Compute congestion term via mean interaction kernel
    delta_X = X_n_reshaped.unsqueeze(2) - X_n_reshaped.unsqueeze(1)  # [T, N, N, 2]
    cov_rho_m = obs.rho(delta_X)                                      # [T, N, N]
    cov_rho_m = torch.mean(cov_rho_m, dim=2)                         # [T, N]

    # Compute obstacle cost term f_obs(t, x)
    f_obs = []
    for ti in range(t.shape[0]):
        f_obs_i = obs.F_obstacle_func_loss(X_n_reshaped[ti])          # [N, 1]
        f_obs.append(f_obs_i)
    f_obs = torch.stack(f_obs, dim=0).squeeze(-1)                    # [T, N]

    # Compute control cost u = -∇V(t, x)
    u = -V_NN.get_grad(t_expanded, X_n)                              # [T*N, dim]
    u = u.reshape(t.shape[0], x.shape[0], obs.dim)                    # [T, N, dim]
    control_cost = obs.c * torch.sum(u * u, dim=2)                    # [T, N]

    # Running cost: L = c‖u‖² + γ_obst·f_obs + γ_cong·ρ
    l = control_cost + obs.gamma_obst * f_obs + obs.gamma_cong * cov_rho_m  # [T, N]

    # Terminal cost
    g = obs.psi_func(X_n_reshaped[-1])                                # [N, 1] or [N]
    
    # Total cost: average over all agents
    J = torch.mean(torch.sum(l[:-1], dim=0) / (t.shape[0] - 1) + g.squeeze())

    return J.item()



def Comp_V(x, V_NN):
    """
    Compute the initial value function V(0, x) averaged over the population.

    Args:
        x     : Initial samples [N, dim]
        V_NN  : Value function network

    Returns:
        V0 (float) : Average value at time t = 0
    """
    t0 = torch.zeros_like(x[:, :1])   # Shape [N, 1]
    V0 = torch.mean(V_NN(t0, x))      # Scalar mean

    return V0.item()


#==================================Wasserstein==============================#

def wasserstein_fp(obs, G_NN_list,              # BR passés (longueur k)
                   G_NN_new,               # BR courant   (G_k)
                   x0, m0,                    # (N,d)  échantillon initial
                   t_grid,                 # (T,1)  instants
                   p          = 1,
                   blur       = 0.05,
                   device     = "cpu"):
    """
    W_p(  m̄_k(t) , BR_k(t) )  pour t ∈ t_grid,   avec
        m̄_k = (1/(k+1)) [ δ_{x0} + Σ_{i=0}^{k-1} δ_{G_i} ].
    Retourne  np.ndarray shape (T,)  des distances.
    """

    T       = t_grid.shape[0]
    N, d    = x0.shape
    k       = len(G_NN_list)
    loss_fn = SamplesLoss("sinkhorn", p=p, blur=blur, backend="tensorized")

    # ---------- pré-calcul des trajectoires BR_k -------------------
    with torch.no_grad():
        t_big = t_grid.repeat_interleave(N, dim=0)          # (T*N,1)
        x_rep = x0.repeat(T, 1)                             # (T*N,d)
        br_k  = G_NN_new(t_big, x_rep).view(T, N, d)        # (T,N,d)

    # ---------- pré-calcul des trajectoires passées ----------------
    past_traj = []  
    for G in G_NN_list:
        with torch.no_grad():
            past = G(t_big, x_rep).view(T, N, d)
        past_traj.append(past)
    if past_traj:
        past_traj = torch.stack(past_traj, dim=0)           # (k,T,N,d)

    # ---------- distances -----------------------------------------
    dists = torch.empty(T, device="cpu")
    for j in range(T):
        # population : x0 + BR passés (chacun N pts)
        if obs.Round == 0:
            parts = [m0]                                        # (N,d)
        else:
            parts = [] 
        if k:
            parts.append(past_traj[:, j, :, :].reshape(k*N, d))
        X = torch.cat(parts, dim=0)                         # ((k+1)*N,d)
        Y = br_k[j]                                         # (N,d)
        dists[j] = loss_fn(X, Y).cpu()

    return dists.numpy()        # shape (T,)
