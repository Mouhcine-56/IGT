import torch
import time
import gc
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
import torch.optim as optim
import copy
from scipy.stats import ks_2samp
import torch.nn as nn
import warnings
import ot
from geomloss import SamplesLoss


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

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

    # Sample domain points
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
    clear_gpu_memory()
    return V_NN

def Solve_MFG(obs, V_NN, G_NN, num_epoch, t, lr, num_samples, device):
    """
    Joint training of V and G (MFG initialization).
    Based on the Solve_MFG2 approach from the reference implementation.
    """
    V_NN.train()
    G_NN.train()
    
    optimizer_V = optim.Adam(V_NN.parameters(), lr=lr)
    optimizer_G = optim.Adam(G_NN.parameters(), lr=lr)
    
    print('Starting Joint Training of V and G (MFG)...')
    
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32, device=device)
    t = t.view(-1, 1)

    for epoch in range(num_epoch + 1):
        # Sample initial conditions
        x0_batch = obs.gen_x0(num_samples, Torch=True)
        
        # Sample domain points
        x_domain = obs.sample_x0(num_samples).requires_grad_(True)
        
        N_t = t.shape[0]
        N_x = num_samples
        
        
        t_in = t.repeat_interleave(N_x).view(-1, 1).requires_grad_(True)
  
        x0_in = x0_batch.repeat(N_t, 1)
        x_domain_in = x_domain.repeat(N_t, 1).requires_grad_(True)
        
        # --- 1. Train V (HJB) ---
        X_t = G_NN(t_in, x0_in)
        X_t_det = X_t.detach().requires_grad_(True)
        
        # 1.1 HJB on Trajectories
        V_pred = V_NN(t_in, X_t_det)
        
        V_t = torch.autograd.grad(outputs=V_pred, inputs=t_in,
                                  grad_outputs=torch.ones_like(V_pred),
                                  create_graph=True, retain_graph=True)[0]
        
        V_x = torch.autograd.grad(outputs=V_pred, inputs=X_t_det,
                                  grad_outputs=torch.ones_like(V_pred),
                                  create_graph=True, retain_graph=True)[0]
        # Congestion term

        cong_term = obs.rho(X_t_det[:, :2].view(N_t, N_x, 2).unsqueeze(2) - 
                          X_t_det[:, :2].view(N_t, N_x, 2).unsqueeze(1))
        cong_term = torch.mean(cong_term, dim=2).view(-1, 1)
        
        # Obstacle term
        f_obs = obs.F_obstacle_func_loss(X_t_det)
        
        # Hamiltonian
        Ham = -obs.c * torch.sum(V_x**2, dim=1, keepdim=True) + obs.gamma_obst * f_obs + obs.gamma_cong * cong_term
        
        Loss_HJB_traj = torch.mean((V_t + Ham)**2)

        # 1.2 HJB on Domain
        V_pred_dom = V_NN(t_in, x_domain_in)
        
        V_t_dom = torch.autograd.grad(outputs=V_pred_dom, inputs=t_in,
                                  grad_outputs=torch.ones_like(V_pred_dom),
                                  create_graph=True, retain_graph=True)[0]
        
        V_x_dom = torch.autograd.grad(outputs=V_pred_dom, inputs=x_domain_in,
                                  grad_outputs=torch.ones_like(V_pred_dom),
                                  create_graph=True, retain_graph=True)[0]
        
        # Congestion on Domain (population is X_t_det)
        X_pop = X_t_det[:, :2].view(N_t, N_x, 2)
        X_eval = x_domain_in[:, :2].view(N_t, N_x, 2)
        
        Diff = X_eval.unsqueeze(2) - X_pop.unsqueeze(1)
        cong_term_dom = obs.rho(Diff)
        cong_term_dom = torch.mean(cong_term_dom, dim=2).view(-1, 1)
        
        f_obs_dom = obs.F_obstacle_func_loss(x_domain_in)
        
        Ham_dom = -obs.c * torch.sum(V_x_dom**2, dim=1, keepdim=True) + obs.gamma_obst * f_obs_dom + obs.gamma_cong * cong_term_dom
        
        Loss_HJB_dom = torch.mean((V_t_dom + Ham_dom)**2)
        
        Loss_HJB =  Loss_HJB_traj + Loss_HJB_dom
        
        optimizer_V.zero_grad()
        Loss_HJB.backward()
        optimizer_V.step()
        
        # --- 2. Train G (ODE) ---
        X_t = G_NN(t_in, x0_in)
        
        G_t = G_NN.grad_t(t_in, x0_in)
        
        V_pred = V_NN(t_in, X_t)
        V_x = torch.autograd.grad(outputs=V_pred, inputs=X_t,
                                  grad_outputs=torch.ones_like(V_pred),
                                  create_graph=True, retain_graph=True)[0]
        
        # ODE Loss: dX/dt = 2c * (-V_x)
        Loss_ODE = torch.mean((G_t - 2 * obs.c * (-V_x))**2)
        
        optimizer_G.zero_grad()
        Loss_ODE.backward()
        optimizer_G.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:5d}: Loss_HJB = {Loss_HJB.item():.4e}, Loss_ODE = {Loss_ODE.item():.4e}")
        
        # Periodic cleanup every 2000 epochs
        if epoch % 2000 == 0 and epoch > 0:
            clear_gpu_memory()
    
    # Final cleanup
    del x0_batch, x_domain, t_in, x0_in, x_domain_in, X_t, X_t_det
    clear_gpu_memory()
            
    print('\n')
    return V_NN, G_NN

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

    # Sample domain points
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
        Loss_total = 0.5 * Loss_hjb + Loss_v + Loss_v_x

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
    clear_gpu_memory()
    return V_NN



def generate_data(obs, V_NN, num_samples, t, lr, num_samples_hjb, device, num_epoch=1000, num_workers=4):
    """
    Generate BVP data in parallel using ThreadPoolExecutor.
    Matches the behavior of the sequential version.
    Memory-optimized: reduced workers to avoid OOM with max_nodes=5000.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Force garbage collection before starting
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    def make_eval_u(V_NN_ref):
        """Create eval_u closure with current V_NN reference."""
        def eval_u(t, x):
            u = -V_NN_ref.get_grad(t, x).detach().cpu().numpy()
            return u
        return eval_u
    
    def make_bvp_guess(V_NN_ref):
        """Create bvp_guess closure with current V_NN reference."""
        def bvp_guess(t, x):
            V_NN_ref.eval()
            V = V_NN_ref(torch.tensor(t, dtype=torch.float32, device=device), 
                     torch.tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy() 
            V_x = V_NN_ref.get_grad(t, x).detach().cpu().numpy() 
            return V, V_x
        return bvp_guess

    def solve_single_bvp(args):
        """Solve a single BVP for initial condition X0."""
        X0, eval_u, bvp_guess = args
        try:
            t_start = 0.0
            
            # Integrate closed-loop system
            SOL = solve_ivp(obs.dynamics, [t_start, obs.TT], X0,
                            method=obs.ODE_solver,
                            args=(eval_u,),
                            rtol=1e-08) 
            
            V_guess, A_guess = bvp_guess(SOL.t.reshape(1, -1).T, SOL.y.T)
            
            # Solve BVP
            X_aug_guess = np.vstack((SOL.y, A_guess.T, V_guess.T))
            SOL_bvp = solve_bvp(obs.aug_dynamics, obs.make_bc(X0), SOL.t, X_aug_guess,
                                verbose=0,
                                tol=obs.data_tol,  
                                max_nodes=obs.max_nodes)
            
            # Free memory from IVP solution
            del SOL, V_guess, A_guess, X_aug_guess
            
            if SOL_bvp.success:
                result = {
                    't': SOL_bvp.x.reshape(1, -1).T.copy(),
                    'X': SOL_bvp.y[:obs.dim].T.copy(),
                    'A': SOL_bvp.y[obs.dim:2*obs.dim].T.copy(),
                    'V': (SOL_bvp.y[-1:] + obs.terminal_cost(SOL_bvp.y[:obs.dim, -1])).T.copy(),
                    'success': True
                }
                del SOL_bvp
                return result
            else:
                msg = SOL_bvp.message
                del SOL_bvp
                return {'success': False, 'message': msg}
                
        except Exception as e:
            return {'success': False, 'message': str(e)}

    print('Generating data (parallel)...')

    dim = obs.dim
    start_time = time.time()
    x0_int = obs.gen_x0(num_samples * 3, Torch=False)  # Generate extra for failures

    X_OUT = np.empty((0, dim))
    A_OUT = np.empty((0, dim))
    V_OUT = np.empty((0, 1))
    t_OUT = np.empty((0, 1))

    Ns_sol = 0
    idx = 0
    failure_count = 0
    hjb_retrain_count = 0
    max_hjb_retrains = 5
    
    # Create initial closures with current V_NN
    eval_u = make_eval_u(V_NN)
    bvp_guess = make_bvp_guess(V_NN)

    while Ns_sol < num_samples and idx < len(x0_int):
        # Determine batch size
        remaining_samples = num_samples - Ns_sol
        remaining_x0 = len(x0_int) - idx
        batch_size = min(num_workers, remaining_samples, remaining_x0)
        
        if batch_size <= 0:
            break
        
        # Prepare batch with current closures
        batch_args = [(x0_int[idx + i, :], eval_u, bvp_guess) for i in range(batch_size)]
        
        print(f'Solving BVP batch ({Ns_sol}/{num_samples} done, idx={idx})...', end='\r')
        
        batch_success = 0
        batch_fail = 0
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(solve_single_bvp, args) for args in batch_args]
            
            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    t_OUT = np.vstack((t_OUT, result['t']))
                    X_OUT = np.vstack((X_OUT, result['X']))
                    A_OUT = np.vstack((A_OUT, result['A']))
                    V_OUT = np.vstack((V_OUT, result['V']))
                    Ns_sol += 1
                    batch_success += 1
                else:
                    batch_fail += 1
                    failure_count += 1
                # Free result memory
                del result
        
        # Free batch memory after each batch
        del batch_args, futures
        gc.collect()
        
        idx += batch_size
        
        # Check if we need to retrain HJB (like original: retrain if Ns_sol < 45 and failures)
        if batch_fail > 0 and Ns_sol < 45 and hjb_retrain_count < max_hjb_retrains:
            # High failure rate in this batch
            if batch_fail >= batch_size // 2:  # More than half failed
                print(f"\nHigh failure rate ({batch_fail}/{batch_size}), Ns_sol={Ns_sol}, retraining HJB...")
                V_NN = Solve_HJB(obs, V_NN, num_epoch, t, lr, num_samples_hjb, device)
                # Recreate closures with updated V_NN
                eval_u = make_eval_u(V_NN)
                bvp_guess = make_bvp_guess(V_NN)
                hjb_retrain_count += 1
        
        # If we've used all x0 but still need more samples, regenerate
        if idx >= len(x0_int) and Ns_sol < num_samples:
            print(f"\nRegenerating initial conditions (have {Ns_sol}/{num_samples})...")
            x0_int = obs.gen_x0(num_samples * 3, Torch=False)
            idx = 0
            
            # Also retrain HJB if we haven't reached threshold
            if Ns_sol < 45 and hjb_retrain_count < max_hjb_retrains:
                print("Retraining HJB before retry...")
                V_NN = Solve_HJB(obs, V_NN, num_epoch, t, lr, num_samples_hjb, device)
                eval_u = make_eval_u(V_NN)
                bvp_guess = make_bvp_guess(V_NN)
                hjb_retrain_count += 1

    print(f'\nGenerated {X_OUT.shape[0]} data from {Ns_sol} BVP solutions in {time.time() - start_time:.1f} sec')
    print(f'Total failures: {failure_count}, HJB retrains: {hjb_retrain_count}\n')

    data = {'t': torch.tensor(t_OUT, dtype=torch.float32, device=device), 
            'X': torch.tensor(X_OUT, dtype=torch.float32, device=device), 
            'A': torch.tensor(A_OUT, dtype=torch.float32, device=device), 
            'V': torch.tensor(V_OUT, dtype=torch.float32, device=device)}

    clear_gpu_memory()
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

        # Weighted total loss (dynamics regularization)
        # Weight can be adjusted based on problem dimension
        ode_weight = 0.01 / obs.dim  # Scale down for higher dimensions
        Loss_total = Loss_G + ode_weight * Loss_ode

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
    # Cleanup
    del t_train, X_train, X_OUT, x_rand
    clear_gpu_memory()
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
    #V_NN = Solve_HJB(an, V_NN, num_epoch_hjb, t, lr_v, num_samples_hjb, device)
    

    # Step 2: Generating
    attempts = 0
    Ns_sol = 0

    while Ns_sol < 45 and attempts < 2:
        N_sol_bvp = 0

        while N_sol_bvp < 1:
            data, Ns_sol = generate_data(an, V_NN, num_samples_bvp, t, lr_v, num_samples_hjb, device)
            N_sol_bvp = Ns_sol

            if N_sol_bvp == 0:
                print(f"Re-solving V{VV} via HJB due to insufficient BVP data...")
                V_NN = Solve_HJB(an, V_NN, num_epoch_hjb, t, lr_v, num_samples_hjb, device)

        # Step 3: Training 
        print(f"\nTraining V{VV} via BVP Approximation...\n")
        V_NN = Approximate_v(an, V_NN, data, num_epochs, t, lr_v, num_samples_hjb, round_num, device)
        
        # Free BVP data after training
        del data
        clear_gpu_memory()
        
        attempts += 1

    return V_NN

    
    
#==================================Compute J and V ==============================#


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
    dim = obs.dim
    T = t.shape[0]
    N = x.shape[0]
    
    # Expand (t, x) to shape [T*N, 1] and [T*N, dim] for G_NN input
    t_expanded = t.repeat_interleave(N, dim=0)              # [T*N, 1]
    x0_expanded = x.repeat(T, 1)                            # [T*N, dim]

    # Simulate trajectories using latest generator
    X_n = obs.G_NN_list[-1](t_expanded, x0_expanded)        # [T*N, dim]
    X_n_reshaped = X_n.reshape(T, N, dim)                   # [T, N, dim]
    X_n_2d = X_n_reshaped[:, :, :2]                         # [T, N, 2] for congestion/obstacles

    # Compute congestion term via mean interaction kernel (only first 2 dims)
    delta_X = X_n_2d.unsqueeze(2) - X_n_2d.unsqueeze(1)     # [T, N, N, 2]
    cov_rho_m = obs.rho(delta_X)                            # [T, N, N]
    cov_rho_m = torch.mean(cov_rho_m, dim=2)                # [T, N]

    # Compute obstacle cost term f_obs(t, x) - uses first 2 dims internally
    f_obs = []
    for ti in range(T):
        f_obs_i = obs.F_obstacle_func_loss(X_n_reshaped[ti])  # [N, 1] - takes full dim, uses [:, :2]
        f_obs.append(f_obs_i)
    f_obs = torch.stack(f_obs, dim=0).squeeze(-1)           # [T, N]

    # Compute control cost u = -∇V(t, x) - all dimensions
    u = -V_NN.get_grad(t_expanded, X_n)                     # [T*N, dim]
    u = u.reshape(T, N, dim)                                # [T, N, dim]
    control_cost = obs.c * torch.sum(u * u, dim=2)          # [T, N]

    # Running cost: L = c‖u‖² + γ_obst·f_obs + γ_cong·ρ
    l = control_cost + obs.gamma_obst * f_obs + obs.gamma_cong * cov_rho_m  # [T, N]

    # Terminal cost - uses full dim now
    g = obs.psi_func(X_n_reshaped[-1])                      # [N, 1]
    
    # Total cost: average over all agents
    J = torch.mean(torch.sum(l[:-1], dim=0) / (T - 1) + g.squeeze())
    
    J_val = J.item()
    
    # Cleanup large tensors
    del t_expanded, x0_expanded, X_n, X_n_reshaped, X_n_2d, delta_X, cov_rho_m, u
    clear_gpu_memory()

    return J_val



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

def wasserstein_fp(obs, G_NN_list,              # Past best responses (length k)
                   G_NN_new,               # Current best response (G_k)
                   x0, m0,                 # (N,d) initial samples
                   t_grid,                 # (T,1) time points
                   p          = 1,
                   blur       = 0.05,
                   device     = "cpu"):
    """
    W_p( m_bar_k(t), BR_k(t) ) for t in t_grid, where
        m_bar_k = (1/(k+1)) [ delta_{x0} + sum_{i=0}^{k-1} delta_{G_i} ].
    Returns np.ndarray of shape (T,) with the distances.
    
    Note: Uses only first 2 dimensions for Wasserstein computation
    since the MFG dynamics only depend on (x1, x2).
    """

    T       = t_grid.shape[0]
    N       = x0.shape[0]
    d       = obs.dim
    k       = len(G_NN_list)
    loss_fn = SamplesLoss("sinkhorn", p=p, blur=blur, backend="tensorized")

    # ---------- Precompute BR_k trajectories -------------------
    with torch.no_grad():
        t_big = t_grid.repeat_interleave(N, dim=0)          # (T*N,1)
        x_rep = x0.repeat(T, 1)                             # (T*N,d)
        br_k  = G_NN_new(t_big, x_rep)[:, :2].view(T, N, 2) # (T,N,2) - only first 2 dims

    # ---------- Precompute past trajectories ----------------
    past_traj = []  
    for G in G_NN_list:
        with torch.no_grad():
            past = G(t_big, x_rep)[:, :2].view(T, N, 2)     # (T,N,2)
        past_traj.append(past)
    if past_traj:
        past_traj = torch.stack(past_traj, dim=0)           # (k,T,N,2)

    # ---------- Distances -----------------------------------------
    dists = torch.empty(T, device="cpu")
    for j in range(T):
        # population: x0[:, :2] + past BRs (N points each)
        if obs.Round == 0:
            parts = [m0[:, :2]]                              # (N,2)
        else:
            parts = [] 
        if k:
            parts.append(past_traj[:, j, :, :].reshape(k*N, 2))
        X = torch.cat(parts, dim=0)                         # ((k+1)*N,2)
        Y = br_k[j]                                         # (N,2)
        dists[j] = loss_fn(X, Y).cpu()

    # Cleanup
    del t_big, x_rep, br_k, past_traj
    clear_gpu_memory()
    
    return dists.numpy()        # shape (T,)