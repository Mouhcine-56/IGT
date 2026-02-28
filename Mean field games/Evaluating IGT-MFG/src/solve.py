import multiprocessing
import torch
import time
import gc
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
import torch.optim as optim
import copy
from scipy.stats import ks_2samp
import torch.nn as nn
from scipy.stats import wasserstein_distance
import ot
from geomloss import SamplesLoss
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
import sys
import os

#================================ DGM_HJB =========================================#

def Solve_HJB(an, V_NN, num_epoch, t, lr, num_samples, device):
   
    V_NN.train()
    optimizer = optim.Adam(V_NN.parameters(), lr)

    x_rand = an.sample_x0(num_samples).requires_grad_(True)

    old_loss = 1
    loss = []  

    for epoch in range(num_epoch+1):

        t = t.requires_grad_(True)

        V_nn =  V_NN(t, x_rand)

        V_nn_t = torch.autograd.grad(outputs=V_nn, inputs=t,
                                          grad_outputs=torch.ones_like(V_nn),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]

        V_nn_x = torch.autograd.grad(outputs=V_nn, inputs=x_rand,
                                          grad_outputs=torch.ones_like(V_nn),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]


        Loss = torch.mean(( V_nn_t + an.ham(t,x_rand,V_nn_x))**2) #+  torch.mean((V_NN(T, x_rand)-an.psi_func(x_rand))**2)


        optimizer.zero_grad()


        Loss.backward()


        optimizer.step()


        if epoch % 1000 == 0:
        #loss.append(old_loss)    
        #new_loss = Loss.item()
            print(f"Iteration {epoch}: Loss = {Loss.item():.4e}")
        #if new_loss>min(loss):
        #    x_rand = x_rand
        #else:   
        #    x_rand = an.sample_x0(num_samples).requires_grad_(True)
        #old_loss = new_loss    

    print('\n') 
    return V_NN

def Solve_MFG1(an, V_NN, G_NN, num_epoch, t, lr, num_samples, device):
    
    V_NN.train()
    G_NN.train()
    
    optimizer_V = optim.Adam(V_NN.parameters(), lr)
    optimizer_G = optim.Adam(G_NN.parameters(), lr)
    
    print('Starting Joint Training of V and G (MFG)...')
    
    # Ensure t is correct shape
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32, device=device)
    t = t.view(-1, 1)

    for epoch in range(num_epoch+1):
        
        # --- Resample x0 every iteration to avoid overfitting ---
        x0_batch = an.gen_x0(num_samples, Torch=True) # (N_x, dim)
        
        
        # Prepare inputs
        N_t = t.shape[0]
        N_x = x0_batch.shape[0]
        
        # Expand t and x0 to evaluate G at all (t, x0) pairs
        # t_in: (N_t * N_x, 1)
        # x0_in: (N_t * N_x, dim)
        t_in = t.repeat_interleave(N_x).view(-1, 1).requires_grad_(True)
        x0_in = x0_batch.repeat(N_t, 1) 
        
        # --- 1. Train V (HJB) ---
        # Forward G (detached, so V training doesn't affect G)
        X_t = G_NN(t_in, x0_in)
        X_t_det = X_t.detach().requires_grad_(True)
        
        # Compute Mean on detached X_t for the coupling term F
        X_reshaped = X_t_det.view(N_t, N_x, -1)
        mean_t = torch.mean(X_reshaped, dim=1) # (N_t, dim)
        mean_t_exp = mean_t.repeat_interleave(N_x, dim=0) # (N_t * N_x, dim)
        
        # Forward V
        V_pred = V_NN(t_in, X_t_det)
        
        # Gradients of V
        V_t = torch.autograd.grad(outputs=V_pred, inputs=t_in,
                                  grad_outputs=torch.ones_like(V_pred),
                                  create_graph=True, retain_graph=True)[0]
        
        V_x = torch.autograd.grad(outputs=V_pred, inputs=X_t_det,
                                  grad_outputs=torch.ones_like(V_pred),
                                  create_graph=True, retain_graph=True)[0]
        
        # HJB Loss
        # F = 0.5 * (x - mean)^2
        F_val = 0.5 * torch.sum((X_t_det - mean_t_exp)**2, dim=1, keepdim=True)
        # Hamiltonian: -0.5 * |p|^2 + F
        Ham = -0.5 * torch.sum(V_x**2, dim=1, keepdim=True) + F_val
        
        Loss_HJB = torch.mean((V_t + Ham)**2)
        
        optimizer_V.zero_grad()
        Loss_HJB.backward()
        optimizer_V.step()
        
        # --- 2. Train G (ODE) ---
        # Forward G (attached, so G training works)
        X_t = G_NN(t_in, x0_in)
        
        # dG/dt
        G_t = torch.autograd.grad(outputs=X_t, inputs=t_in,
                                  grad_outputs=torch.ones_like(X_t),
                                  create_graph=True, retain_graph=True)[0]
        
        # V_x (attached to G via X_t input)
        # We use V to guide G, but we don't update V here.
        V_pred = V_NN(t_in, X_t)
        V_x = torch.autograd.grad(outputs=V_pred, inputs=X_t,
                                  grad_outputs=torch.ones_like(V_pred),
                                  create_graph=True, retain_graph=True)[0]
        
        # ODE Loss: dX/dt = - V_x
        Loss_ODE = torch.mean((G_t + V_x)**2)
        
        optimizer_G.zero_grad()
        Loss_ODE.backward()
        optimizer_G.step()
        
        if epoch % 1000 == 0:
            print(f"Iteration {epoch}: Loss_HJB = {Loss_HJB.item():.4e}, Loss_ODE = {Loss_ODE.item():.4e}")
            
    print('\n')
    return V_NN, G_NN

def Solve_MFG2(an, V_NN, G_NN, num_epoch, t, lr, num_samples, device):
    
    V_NN.train()
    G_NN.train()
    
    optimizer_V = optim.Adam(V_NN.parameters(), lr)
    optimizer_G = optim.Adam(G_NN.parameters(), lr)
    
    print('Starting Joint Training of V and G (MFG)...')
    
    # Ensure t is correct shape
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32, device=device)
    t = t.view(-1, 1)

    for epoch in range(num_epoch+1):
        
        # --- 1. Samples for Population (G) ---
        # Sample initial conditions from m0 for the generator (to compute Mean Field correctly)
        x0_batch = an.gen_x0(num_samples, Torch=True) # (N_x, dim)
        
        # --- 2. Samples for Value Function (V) ---
        # Sample uniformly from the domain to learn global V
        x_domain = an.sample_x0(num_samples).requires_grad_(True) # (N_x, dim)
        
        # Prepare inputs
        N_t = t.shape[0]
        N_x = num_samples
        
        # Expand t
        t_in = t.repeat_interleave(N_x).view(-1, 1).requires_grad_(True)
        
        # Expand x0 for G
        x0_in = x0_batch.repeat(N_t, 1)
        
        # Expand x_domain for V
        # We repeat x_domain for each time step to cover [0,T] x Domain
        x_domain_in = x_domain.repeat(N_t, 1).requires_grad_(True)
        
        # --- Compute Mean Field from G ---
        # We need the population mean at each time t to compute F in HJB
        X_t = G_NN(t_in, x0_in)
        X_t_det = X_t.detach() # Detach for mean calculation
        
        X_reshaped = X_t_det.view(N_t, N_x, -1)
        mean_t = torch.mean(X_reshaped, dim=1) # (N_t, dim)
        mean_t_exp = mean_t.repeat_interleave(N_x, dim=0) # (N_t * N_x, dim)
        
        # --- 1. Train V (HJB) on Domain ---
        # Evaluate V on uniform domain samples
        V_pred = V_NN(t_in, x_domain_in)
        
        # Gradients of V
        V_t = torch.autograd.grad(outputs=V_pred, inputs=t_in,
                                  grad_outputs=torch.ones_like(V_pred),
                                  create_graph=True, retain_graph=True)[0]
        
        V_x = torch.autograd.grad(outputs=V_pred, inputs=x_domain_in,
                                  grad_outputs=torch.ones_like(V_pred),
                                  create_graph=True, retain_graph=True)[0]
        
        # HJB Loss
        # F = 0.5 * (x - mean)^2
        # Note: mean_t_exp corresponds to time t, which matches x_domain_in's time component

        F_val = 0.5 * torch.sum((x_domain_in - mean_t_exp)**2, dim=1, keepdim=True)

        # Hamiltonian: -0.5 * |p|^2 + F
        Ham = -0.5 * torch.sum(V_x**2, dim=1, keepdim=True) + F_val

        Loss_HJB = torch.mean((V_t + Ham)**2)
        
        optimizer_V.zero_grad()
        Loss_HJB.backward()
        optimizer_V.step()
        
        # --- 2. Train G (ODE) ---
        # Forward G (attached)
        X_t = G_NN(t_in, x0_in)
        
        # dG/dt
        G_t = G_NN.grad_t(t_in, x0_in)
        
        # V_x at G's location (to guide G)
        V_pred_G = V_NN(t_in, X_t)
        V_x_G = torch.autograd.grad(outputs=V_pred_G, inputs=X_t,
                                  grad_outputs=torch.ones_like(V_pred_G),
                                  create_graph=True, retain_graph=True)[0]
        
        
        # ODE Loss: dX/dt = - V_x
        Loss_ODE = torch.mean((G_t + V_x_G)**2)
        
        optimizer_G.zero_grad()
        Loss_ODE.backward() 
        optimizer_G.step()
        
        if epoch % 1000 == 0:
            print(f"Iteration {epoch}: Loss_HJB = {Loss_HJB.item():.4e}, Loss_ODE = {Loss_ODE.item():.4e}")
            
    # Cleanup after training
    del X_t, V_pred, V_t, V_x, G_t, V_x_G, Loss_HJB, Loss_ODE
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
            
    print('\n')
    return V_NN, G_NN

def Solve_MFG3(an, V_NN, G_NN, num_epoch, t, lr, num_samples, device):
    
    V_NN.train()
    G_NN.train()
    
    # Reduced learning rate for stability in high dimension
    optimizer_V = optim.Adam(V_NN.parameters(), lr=lr)
    optimizer_G = optim.Adam(G_NN.parameters(), lr=lr)
    
    print(f'Starting Joint Training of V and G (MFG) in dim {an.dim}...')
    
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32, device=device)
    t = t.view(-1, 1)

    for epoch in range(num_epoch+1):
        
        # --- 1. Sampling ---
        # For G: Real initial conditions (Gaussian)
        x0_batch = an.gen_x0(num_samples, Torch=True) # (N_x, dim)
        
        # For V: Sample AROUND the current population to learn V where it is useful
        # Mixture of uniform points and points near the trajectory
        if epoch < 500:
             # Early on, broad exploration
             x_domain = an.sample_x0(num_samples).requires_grad_(True)
        else:
             # Then concentrate around the theoretical mean (mu=0.1) with broader variance
             # This avoids learning V in regions far from the occupied space
             noise = torch.randn(num_samples, an.dim, device=device) * 2.0 # Sigma=2.0 for broad coverage
             x_domain = (torch.tensor(an.mu, device=device) + noise).float().requires_grad_(True)

        N_t = t.shape[0]
        N_x = num_samples
        
        # Expand t
        t_in = t.repeat_interleave(N_x).view(-1, 1).requires_grad_(True)
        
        # Expand x0 for G
        x0_in = x0_batch.repeat(N_t, 1)
        
        # Expand x_domain for V
        x_domain_in = x_domain.repeat(N_t, 1).requires_grad_(True)
        
        # --- 2. Compute the Mean Field ---
        # Use G to estimate the population mean at each time point
        X_t = G_NN(t_in, x0_in)
        X_t_det = X_t.detach() 
        
        X_reshaped = X_t_det.view(N_t, N_x, -1)
        mean_t = torch.mean(X_reshaped, dim=1) # (N_t, dim)
        mean_t_exp = mean_t.repeat_interleave(N_x, dim=0) # (N_t * N_x, dim)
        
        # --- 3. Train V (HJB) ---
        V_pred = V_NN(t_in, x_domain_in)
        
        grad_outputs = torch.ones_like(V_pred)
        V_t = torch.autograd.grad(V_pred, t_in, grad_outputs=grad_outputs, create_graph=True)[0]
        V_x = torch.autograd.grad(V_pred, x_domain_in, grad_outputs=grad_outputs, create_graph=True)[0]
        
        # HJB Loss
        # F = 0.5 * |x - m|^2
        # IMPORTANT: In dim 10, this value can blow up. Could normalize by dimension if needed,
        # but here we keep the physical definition.
        diff = x_domain_in - mean_t_exp
        F_val = 0.5 * torch.sum(diff**2, dim=1, keepdim=True)
        
        # Hamiltonian: -0.5 * |p|^2 + F
        # p = V_x
        Ham = -0.5 * torch.sum(V_x**2, dim=1, keepdim=True) + F_val
        
        # HJB residual
        res_hjb = V_t + Ham
        
        # Terminal penalty (optional but helps convergence)
        # If t=T, V(T,x) should be close to 0 (terminal cost = 0)
        # Not added explicitly here since t covers [0, T], but it is implicit
        
        Loss_HJB = torch.mean(res_hjb**2)
        
        optimizer_V.zero_grad()
        Loss_HJB.backward()
        # Gradient clipping to avoid explosion in high dimension
        torch.nn.utils.clip_grad_norm_(V_NN.parameters(), 1.0)
        optimizer_V.step()
        
        # --- 4. Train G (ODE) ---
        # Re-evaluate G to have the computation graph
        X_t_G = G_NN(t_in, x0_in)
        
        G_t = torch.autograd.grad(X_t_G, t_in, grad_outputs=torch.ones_like(X_t_G), create_graph=True)[0]
        
        # We want dX/dt = -nabla V(t, X)
        # Evaluate V at point X_t_G (the current position of particles)
        V_pred_G = V_NN(t_in, X_t_G)
        V_x_G = torch.autograd.grad(V_pred_G, X_t_G, grad_outputs=torch.ones_like(V_pred_G), create_graph=True)[0]
        
        # Loss ODE
        Loss_ODE = torch.mean((G_t + V_x_G)**2)
        
        optimizer_G.zero_grad()
        Loss_ODE.backward()
        torch.nn.utils.clip_grad_norm_(G_NN.parameters(), 1.0)
        optimizer_G.step()
        
        if epoch % 1000 == 0:
            print(f"Iteration {epoch}: Loss_HJB = {Loss_HJB.item():.4e}, Loss_ODE = {Loss_ODE.item():.4e}")
            
    print('\n')
    return V_NN, G_NN

#================================  BVP + HJB =========================================#

def Approximate_v(an, V_NN, data, num_epoch, t, lr, num_samples, Round, device, weight_decay=0.0):

    V_NN.train()
    optimizer = optim.Adam(V_NN.parameters(), lr=lr, weight_decay=weight_decay)

    x_rand = an.sample_x0(num_samples).requires_grad_(True)

    old_loss = 1
    loss = []


    for epoch in range(num_epoch+1):

        t = t.requires_grad_(True)

        V_nn =  V_NN(t, x_rand)

        V_nn_t = torch.autograd.grad(outputs=V_nn, inputs=t,
                                          grad_outputs=torch.ones_like(V_nn),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]

        V_nn_x = torch.autograd.grad(outputs=V_nn, inputs=x_rand,
                                          grad_outputs=torch.ones_like(V_nn),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]

        Loss_hjb = torch.mean(( V_nn_t + an.ham(t,x_rand,V_nn_x))**2) 
        Loss_v = torch.mean((V_NN(data['t'], data['X']) - data['V'])**2) 
        Loss_v_x = torch.mean((V_NN.get_grad(data['t'], data['X']) - data['A'])**2)

        Loss_total =   0.1*Loss_hjb + Loss_v + Loss_v_x

        optimizer.zero_grad()


        Loss_total.backward()


        optimizer.step()


        if epoch % 1000 == 0:
            loss.append(old_loss)    
            new_loss = Loss_total.item()
            print(f"Iteration {epoch}: Loss_V = {Loss_v.item():.4e}, Loss_V_x = {Loss_v_x.item():.4e}, Loss_HJB = {Loss_hjb.item():.4e}, Loss_total = {Loss_total.item():.4e}")
            if new_loss>min(loss):
               x_rand = x_rand
            else:   
               x_rand = an.sample_x0(num_samples).requires_grad_(True)
            old_loss = new_loss  

    # Cleanup after training
    del V_nn, V_nn_t, V_nn_x, Loss_hjb, Loss_v, Loss_v_x, Loss_total
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print('\n')  
   
    return V_NN



def Approximate_v2(an, V_NN, data, num_epoch, t, lr, num_samples, Round, device, weight_decay=0.0):
   
    V_NN.train()
    optimizer = optim.Adam(V_NN.parameters(), lr=lr, weight_decay=weight_decay)

    x_rand = an.sample_x0(num_samples).requires_grad_(True)

    for epoch in range(num_epoch+1):

        Loss_v = torch.mean((V_NN(data['t'], data['X']) - data['V'])**2) 
        Loss_v_x = torch.mean((V_NN.get_grad(data['t'], data['X']) - data['A'])**2)

        Loss_total =  Loss_v + Loss_v_x

        optimizer.zero_grad()


        Loss_total.backward()


        optimizer.step()


        if epoch % 1000 == 0:

            print(f"Iteration {epoch}: Loss_V = {Loss_v.item():.4e}, Loss_V_x = {Loss_v_x.item():.4e}, Loss_total = {Loss_total.item():.4e}")

    print('\n')      
    return V_NN

#================================ solve BVP ===========================================#

def generate_data(an, V_NN, num_samples, device):
    
    def eval_u(t, x):
        u = - V_NN.get_grad(t, x).detach().cpu().numpy()
        return u
    
    def bvp_guess(t, x):
        V_NN.eval()
        V = V_NN(torch.tensor(t, dtype=torch.float32, device=device), 
                 torch.tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy()
        V_x = V_NN.get_grad(t, x).detach().cpu().numpy()
        return V, V_x

    print('Generating data_OC...')

    dim = an.dim

    X_OUT = np.empty((0, dim))
    A_OUT = np.empty((0, dim))
    V_OUT = np.empty((0, 1))
    t_OUT = np.empty((0, 1))

    Ns_sol = 0
    start_time = time.time()
    x0_int = an.gen_x0(num_samples, Torch=False)
#     data = np.load('data.npz')
#     x0_int = data['x']

    # ----------------------------------------------------------------------

    while Ns_sol < num_samples:
        
        #print('Solving BVP #', Ns_sol+1, '...', end='\r')

        #X0 = an.gen_x0(1).flatten()
        X0 = x0_int[Ns_sol,:]
        bc = an.make_bc(X0)

        # Integrates the closed-loop system (NN controller)

        SOL = solve_ivp(an.dynamics, [0., an.TT], X0,
                        method=an.ODE_solver,
                        args=(eval_u,),
                        rtol=1e-08)

        V_guess, A_guess = bvp_guess(SOL.t.reshape(1,-1).T, SOL.y.T)
        
        #print(SOL.y)

        try:
            # Solves the two-point boundary value problem
            
            X_aug_guess = np.vstack((SOL.y, A_guess.T, V_guess.T))

            SOL = solve_bvp(an.aug_dynamics, bc, SOL.t, X_aug_guess,
                            verbose=0,
                            tol=an.data_tol,
                            max_nodes=an.max_nodes)
            if not SOL.success:
                warnings.warn(Warning())

            Ns_sol += 1

            V = SOL.y[-1:] + an.terminal_cost(SOL.y[:dim,-1])

            t_OUT = np.vstack((t_OUT, SOL.x.reshape(1,-1).T))
            X_OUT = np.vstack((X_OUT, SOL.y[:dim].T))
            A_OUT = np.vstack((A_OUT, SOL.y[dim:2*dim].T))
            V_OUT = np.vstack((V_OUT, V.T))

        except Warning:
            pass

    print('Generated', X_OUT.shape[0], 'data from', Ns_sol,
        'BVP solutions in %.1f' % (time.time() - start_time), 'sec \n')

    data = {'t': torch.tensor(t_OUT, dtype=torch.float32, device=device), 
            'X': torch.tensor(X_OUT, dtype=torch.float32, device=device), 
            'A': torch.tensor(A_OUT, dtype=torch.float32, device=device), 
            'V': torch.tensor(V_OUT, dtype=torch.float32, device=device)}

    return data
   
#==========================   Simulate points   ==========================================#   


def sim_points(an, V_NN, num_samples, N, t0, tf,  device):
    
    def eval_u(t, x):
        u = - V_NN.get_grad(t, x).detach().cpu().numpy()
        return u
    
    # Pre-allocate arrays instead of repeated vstack (more RAM-efficient)
    X_OUT = np.empty((num_samples * (N+1), an.dim))
    t_OUT = np.empty((num_samples * (N+1), 1))
    
    data = an.gen_x0(num_samples, Torch=False)
    
    Ns_sol = 0
    start_time = time.time()
    
    print('Generating data_MFG...')
    
    while Ns_sol < num_samples:
        
        X0 = data[Ns_sol, :]
        
        # Integrates the closed-loop system (NN controller)
        SOL = solve_ivp(an.dynamics, [0., 1], X0,
                        method= 'RK23', t_eval=np.linspace(0,1,N+1),
                        args=(eval_u,),
                        rtol=1e-08)

        # Fill pre-allocated arrays
        idx_start = Ns_sol * (N+1)
        idx_end = (Ns_sol + 1) * (N+1)
        t_OUT[idx_start:idx_end, 0] = SOL.t
        X_OUT[idx_start:idx_end, :] = SOL.y.T
        
        Ns_sol += 1
    
    t_train = t_OUT.reshape(num_samples, N+1)
    t_tr = t_train.T.flatten().reshape(-1, 1)
    x_tr = np.tile(data[0:num_samples, :], (N+1, 1))
    X_OUT = X_OUT.reshape(num_samples, N+1, an.dim)
    X_OUT = X_OUT.transpose(1, 0, 2).reshape((N+1) * num_samples, an.dim)
    
    # Free data after use
    del data
    gc.collect()
    
    print('Generated', X_OUT.shape[0], 'data from', Ns_sol,
        'IVP solutions in %.1f' % (time.time() - start_time), 'sec \n')
    
        
    return t_tr, x_tr, t_OUT, X_OUT


#================================  Train Generator =========================================#

def Train_Gen(an, G_NN, V_NN, t_tr, x_tr, X_OUT, num_epoch, t, lr, num_samples, Round, device):
    
    G_NN.eval()
    
    # Save original weights instead of deepcopy to save memory
    original_state_dict = {k: v.clone() for k, v in G_NN.state_dict().items()}
    
    x_rand = an.gen_x0(num_samples, Torch=True)
    t = t.requires_grad_(True)

    t_train = torch.tensor(t_tr, dtype=torch.float32, device=device).requires_grad_(True)
    X_train = torch.tensor(x_tr, dtype=torch.float32, device=device).requires_grad_(True)
    X_OUT = torch.tensor(X_OUT, dtype=torch.float32, device=device).requires_grad_(True)
    
    
    G_NN.train()
    optimizer = optim.Adam(G_NN.parameters(), lr)
    
    old_loss = 1
    loss = []

    for epoch in range(num_epoch+1):
        
        gen_samples = G_NN(t,x_rand)

        G_nn_t =  G_NN.grad_t(t, x_rand)
       
        Loss_ode = torch.mean(( G_nn_t - an.dynamics_torch(t,G_NN(t, x_rand),V_NN))**2)
        Loss_G = torch.mean((G_NN(t_train, X_train) - X_OUT)**2) 

        Loss_total =  Loss_G   + 0.1*Loss_ode  

        optimizer.zero_grad()

        Loss_total.backward()

        optimizer.step()


        if epoch % 1000 == 0:
            loss.append(old_loss)    
            new_loss = Loss_total.item()
            print(f"Iteration {epoch}:  Loss_G = {Loss_G.item():.4e},  Loss_ODE = {Loss_ode.item():.4e},  Loss_total = {Loss_total.item():.4e}")
            if new_loss>min(loss):
               x_rand = x_rand
            else:   
               x_rand = an.gen_x0(num_samples, Torch=True).requires_grad_(True)
            old_loss = new_loss    
    
    # Save trained state
    trained_state_dict = {k: v.clone() for k, v in G_NN.state_dict().items()}
    
    # Restore G_NN to original weights
    G_NN.load_state_dict(original_state_dict)
    
    # Create G_NN_new with the trained weights
    from model import G_Net
    G_NN_new = G_Net(dim=an.dim, ns=G_NN.ns, act_func=G_NN.act_func, hh=G_NN.hh,
                    device=device, mu=G_NN.mu, std=G_NN.std, TT=G_NN.TT).to(device)
    G_NN_new.load_state_dict(trained_state_dict)
    
    # Cleanup after training
    del gen_samples, G_nn_t, Loss_ode, Loss_G, Loss_total
    del t_train, X_train, X_OUT
    del original_state_dict, trained_state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print('\n')
    return G_NN, G_NN_new



#==================================Compute J and V ==============================#

def Comp_J0(an, t, x, V_NN, G_NN=None):
    """Compute J using G_NN (or an.G_NN_list[-1] if G_NN is None)."""
    t_expanded = t.repeat_interleave(x.shape[0]).view(-1,1)
    x0_expanded = x.repeat(t.shape[0],1)

    # Use G_NN passed as argument, otherwise use an.G_NN_list[-1]
    if G_NN is not None:
        Xn = G_NN(t_expanded, x0_expanded)
    else:
        Xn = an.G_NN_list[-1](t_expanded, x0_expanded)

    mean = torch.mean(Xn.reshape(t.shape[0], x.shape[0], an.dim),
                            dim=1
                        )
    
    mns = Xn.reshape(t.shape[0], x.shape[0], an.dim) - mean.reshape(t.shape[0], 1, an.dim)
    
    F = 0.5 * torch.sum(mns * mns, dim=2)

    u = - V_NN.get_grad(t_expanded, Xn).reshape(t.shape[0], x.shape[0], an.dim)

    l = 0.5 * torch.sum(u * u, dim=2) +  F
    J = torch.mean(1/t.shape[0] * torch.sum(l, dim=0))
    
    return J.item()


def Comp_J1(an, t, x, V_NN, G_NN2):
    
    t_expanded = t.repeat_interleave(x.shape[0]).view(-1,1)
    x0_expanded = x.repeat(t.shape[0],1)
    
    x1 = an.gen_x0(num_samples=x.shape[0], Torch=True)
    x1_expanded = x1.repeat(t.shape[0],1)
    
    Xn = G_NN2(t_expanded, x1_expanded)
    
    dist_0 = an.G_NN_list[-1](t_expanded, x0_expanded)

    mean = torch.mean(Xn.reshape(t.shape[0], x.shape[0], an.dim),
                            dim=1
                        )
    
    mns = Xn.reshape(t.shape[0], x.shape[0], an.dim) - mean.reshape(t.shape[0], 1, an.dim)
    
    F = 0.5 * torch.sum(mns * mns, dim=2)

    u = - V_NN.get_grad(t_expanded, Xn).reshape(t.shape[0], x.shape[0], an.dim)

    l = 0.5 * torch.sum(u * u, dim=2) +  F
    J = torch.mean(1/t.shape[0] * torch.sum(l, dim=0))
    
    return J.item() 


def Comp_V(x, V_NN):
    
    t0 = torch.zeros_like(x[:,0:1])
    V0 = torch.mean(V_NN(t0, x)) 
    
    return V0.item()
    
  

#==================================Update==============================#
def wasserstein_distance_1(model1, model2, t, x0):
    
    
    t_expanded = t.repeat_interleave(x0.shape[0]).view(-1,1)
    x0_expanded = x0.repeat(t.shape[0],1)
    

    M_old = model1(t_expanded, x0_expanded).reshape(t.shape[0], x0.shape[0])
    
    M_new = model2(t_expanded, x0_expanded).reshape(t.shape[0], x0.shape[0])
    
    # Ensure tensors are on CPU and converted to numpy
    M_old_np = M_old.detach().cpu().numpy()
    M_new_np = M_new.detach().cpu().numpy()

    # Compute Wasserstein distances per time step
    distances = []
    for i in range(M_old_np.shape[0]):
        d = wasserstein_distance(M_old_np[i], M_new_np[i])
        distances.append(d)

    distances = np.array(distances)

    # Compute norms of the distance vector
    norm_l1 = np.linalg.norm(distances, ord=1)
    norm_l2 = np.linalg.norm(distances, ord=2)
    norm_linf = np.linalg.norm(distances, ord=np.inf)

    # Print
    print("\n=== Wasserstein Distance Norms over Time ===")
    print(f"L1 norm    : {norm_l1:.4e}")
    print(f"L2 norm    : {norm_l2:.4e}")
    print(f"Linf norm  : {norm_linf:.4e}")
    print("============================================\n")
    
def wasserstein_population_vs_br(an, GNN_list, GNN_new, m0, t, x0, p=1):

    T      = t.shape[0]
    N0, d  = x0.shape
    k      = len(GNN_list)              
    distances = []

    # Loop over time steps
    for ti in t:
        
        # 1) bar(m) from historical generators
        if an.Round==0:
            pop_pts = [m0.cpu()]
        else:
             pop_pts = []
        for G in GNN_list:
            with torch.no_grad():
                ti_b = ti.repeat_interleave(N0 ).view(-1, 1)   # (N0,1)
                pts  = G(ti_b, x0).cpu().numpy()              # (N0,d)
            pop_pts.append(pts)

        X = np.concatenate(pop_pts, axis=0)                   # ((k+1)*N0 , d)
        a = np.ones(X.shape[0]) / X.shape[0]                  # poids uniformes

        # 2) mesure du best-response μ^k
        with torch.no_grad():
            ti_b = ti.repeat_interleave(N0).view(-1, 1)
            Y    = GNN_new(ti_b, x0).cpu().numpy()           # (N0,d)
        b = np.ones(Y.shape[0]) / Y.shape[0]

        # 3) Wasserstein-p
        C   = ot.dist(X, Y, metric='euclidean') #** p
        Wp  = ot.emd2(a, b, C) #** (1.0 / p)
        distances.append(Wp)

    distances = np.array(distances)

    # Compute norms of the distance vector
    norm_l1 = np.linalg.norm(distances, ord=1)
    norm_l2 = np.linalg.norm(distances, ord=2)
    norm_linf = np.linalg.norm(distances, ord=np.inf)

    # Print
    print("\n=== Wasserstein Distance Norms over Time ===")
    print(f"L1 norm    : {norm_l1:.4e}")
    print(f"L2 norm    : {norm_l2:.4e}")
    print(f"Linf norm  : {norm_linf:.4e}")
    print("============================================\n")
    
def wasserstein_population_vs_br_geom(an, GNN_list, GNN_new, m0, t, x0, p=1, blur=0.05):
    """
    Compute Wasserstein-p distances over time between the population distribution (historical generators)
    and the best response, using GeomLoss for efficiency.
    """

    T      = t.shape[0]
    N0, d  = x0.shape
    k      = len(GNN_list)
    device = x0.device

    distances = []

    # Define the Sinkhorn-based loss
    loss_fn = SamplesLoss(loss="sinkhorn", p=p, blur=blur, backend="tensorized")

    for ti in t:
        # 1) Compute population distribution (bar(m))
        if an.Round == 0:
            pop_pts = [m0]
        else:
            pop_pts = []

        for G in GNN_list:
            with torch.no_grad():
                ti_b = ti.repeat_interleave(N0).view(-1, 1).to(device)
                pts = G(ti_b, x0)  # shape (N0, d)
            pop_pts.append(pts)

        X = torch.cat(pop_pts, dim=0)  # shape ((k+1)*N0, d) if Round > 0

        # 2) Best-response samples
        with torch.no_grad():
            ti_b = ti.repeat_interleave(N0).view(-1, 1).to(device)
            Y = GNN_new(ti_b, x0)  # shape (N0, d)

        # 3) Compute Wasserstein distance using GeomLoss
        Wp = loss_fn(X, Y).item()
        distances.append(Wp)

    # Convert to numpy array for norm computations
    distances = np.array(distances)

    # Compute norms
    norm_l1 = np.linalg.norm(distances, ord=1)
    norm_l2 = np.linalg.norm(distances, ord=2)
    norm_linf = np.linalg.norm(distances, ord=np.inf)

    # Print results
    print("\n=== Wasserstein Distance Norms over Time (GeomLoss) ===")
    print(f"L1 norm    : {norm_l1:.4e}")
    print(f"L2 norm    : {norm_l2:.4e}")
    print(f"Linf norm  : {norm_linf:.4e}")
    print("=========================================================\n")
    
def wasserstein_fp(an, G_NN_list,              # Past best responses (length k)
                   G_NN_new,               # Current best response (G_k)
                   x0,                     # (N,d) initial samples
                   t_grid,                 # (T,1) time points
                   p          = 1,
                   blur       = 0.05,
                   device     = "cpu"):
    """
    W_p( m_bar_k(t), BR_k(t) ) for t in t_grid, where
        m_bar_k = (1/(k+1)) [ delta_{x0} + sum_{i=0}^{k-1} delta_{G_i} ].
    Returns np.ndarray of shape (T,) with the distances.

    Memory-optimized version: processes one time step at a time.
    """

    T       = t_grid.shape[0]
    N, d    = x0.shape
    k       = len(G_NN_list)
    
    # Reduce the number of points if too large for memory
    max_points = min(N, 1000)  # Cap at 1000 points for Wasserstein computation
    if N > max_points:
        indices = torch.randperm(N)[:max_points]
        x0_sub = x0[indices]
    else:
        x0_sub = x0
        max_points = N
    
    loss_fn = SamplesLoss("sinkhorn", p=p, blur=blur, backend="tensorized")

    dists = torch.empty(T, device="cpu")
    
    # Process one time step at a time to save memory
    for j in range(T):
        tj = t_grid[j:j+1]  # (1, 1)
        tj_exp = tj.repeat(max_points, 1)  # (max_points, 1)
        
        # Compute BR_k for this time step
        with torch.no_grad():
            Y = G_NN_new(tj_exp, x0_sub)  # (max_points, d)
        
        # Compute past trajectories for this time step
        parts = []
        for G in G_NN_list:
            with torch.no_grad():
                past_pts = G(tj_exp, x0_sub)  # (max_points, d)
            parts.append(past_pts)
        
        if parts:
            X = torch.cat(parts, dim=0)  # (k*max_points, d)
        else:
            # No past G_NN, use x0 as initial distribution
            X = x0_sub
        
        # Compute distance for this time step
        with torch.no_grad():
            dists[j] = loss_fn(X, Y).cpu()
        
        # Free memory after each time step
        del X, Y, parts
        if j % 5 == 0:  # Periodic cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return dists.numpy()        # shape (T,)



#================================ solve BVP (Parallel) ===========================================#

def _solve_single_bvp(args):
    """
    Worker function to solve a single BVP. 
    """
    X0, an_params, V_NN_state, ns, device_str, x0_initial_np = args
    
    import sys
    import os
    import torch
    import numpy as np
    from scipy.integrate import solve_ivp, solve_bvp
    
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        from model import V_Net, G_Net
    except ImportError:
        pass

    device = torch.device(device_str)
    
    dim = an_params['dim']
    TT = an_params['TT']
    Round = an_params['Round']
    n = an_params['n']
    VV = an_params['VV']
    
    x0_initial = torch.tensor(x0_initial_np, dtype=torch.float32, device=device)
    
    # --- 1. Reconstruct G_NN_list ---
    G_NN_list = []
    if 'G_NN_states' in an_params and an_params['G_NN_states']:
        mu = torch.tensor(an_params['mu'], dtype=torch.float32, device=device)
        std = torch.tensor(an_params['std'], dtype=torch.float32, device=device)
        
        for g_state in an_params['G_NN_states']:
            # CORRECTION 1: Utiliser ReLU pour G_Net comme dans main.py
            G_NN = G_Net(
                dim=dim, ns=ns, act_func=lambda x: torch.relu(x), 
                hh=0.5, device=device, mu=mu, std=std, TT=TT
            ).to(device)
            G_NN.load_state_dict(g_state)
            G_NN.eval()
            G_NN_list.append(G_NN)
    
    # --- 2. Reconstruct V_NN ---
    try:
        # V_Net utilise Tanh dans main.py, donc c'est correct ici
        V_NN = V_Net(
            dim=dim, ns=ns, act_func=lambda x: torch.tanh(x),
            hh=0.5, psi_func=lambda x: torch.zeros(x.size(0), device=device),
            device=device, TT=TT
        ).to(device)
        V_NN.load_state_dict(V_NN_state)
        V_NN.eval()
    except Exception as e:
        print(f"Error initializing V_NN: {e}")
        return None

    # --- 3. Define Logic Functions ---

    def update_mean(t_val):
        if np.isscalar(t_val):
            t_val = np.array([t_val])
            
        n_nodes = t_val.shape[0]
        t_tensor = torch.tensor(t_val, dtype=torch.float32, device=device).reshape(-1, 1)
        
        t_expanded = t_tensor.repeat_interleave(x0_initial.shape[0], dim=0)
        x0_expanded = x0_initial.repeat(n_nodes, 1)
        
        if VV == 1:
            Means = []
            with torch.no_grad():
                X_t = G_NN_list[0](t_expanded, x0_expanded)
                mean_new = torch.mean(X_t.reshape(n_nodes, x0_initial.shape[0], dim), dim=1)
                Means.append(mean_new)
            
            if n == 0:
                return Means[0].cpu().numpy()
            else:
                for i in range(1, n + 1):
                    with torch.no_grad():
                        X_t = G_NN_list[i](t_expanded, x0_expanded)
                        mean_new = torch.mean(X_t.reshape(n_nodes, x0_initial.shape[0], dim), dim=1)
                    
                    prev_mean = Means[i-1]
                    updated_mean = (1 / (i + 1)) * mean_new + (i / (i + 1)) * prev_mean
                    Means.append(updated_mean)
                
                return Means[n].cpu().numpy()
        else:
            with torch.no_grad():
                X_t = G_NN_list[-1](t_expanded, x0_expanded)
                mean_new = torch.mean(X_t.reshape(n_nodes, x0_initial.shape[0], dim), dim=1)
            return mean_new.cpu().numpy()

    def eval_u(t_val, x_arr):
        t_tensor = torch.tensor([[t_val]], dtype=torch.float32, device=device)
        x_tensor = torch.tensor(x_arr.reshape(1, -1), dtype=torch.float32, device=device, requires_grad=True)
        
        out = V_NN(t_tensor, x_tensor)
        grad_outputs = torch.ones_like(out)
        grad_inputs = torch.autograd.grad(out, x_tensor, grad_outputs=grad_outputs, create_graph=False)[0]
        
        u = -grad_inputs.detach().cpu().numpy()
        return u.flatten()
    
    def bvp_guess(t_arr, x_arr):
        t_tensor = torch.tensor(t_arr, dtype=torch.float32, device=device)
        x_tensor = torch.tensor(x_arr, dtype=torch.float32, device=device, requires_grad=True)
        
        V = V_NN(t_tensor, x_tensor)
        grad_outputs = torch.ones_like(V)
        V_x = torch.autograd.grad(V, x_tensor, grad_outputs=grad_outputs, create_graph=False)[0]
        
        return V.detach().cpu().numpy(), V_x.detach().cpu().numpy()
    
    def dynamics(t, X, U_fun):
        U = U_fun(t, X)
        return U
    
    def make_bc(X0_in):
        def bc(X_aug_0, X_aug_T):
            X0_bc = X_aug_0[:dim]
            AT = X_aug_T[dim:2*dim]
            vT = X_aug_T[2*dim:]
            dFdXT = 0 
            return np.concatenate((X0_bc - X0_in, AT - dFdXT, vT))
        return bc
    
    def aug_dynamics(t, X_aug):
        Ax = X_aug[dim:2*dim]
        U = -Ax 
        x = X_aug[:dim]
        
        mean = update_mean(t) 
        dFF = (x.T - mean).T 
        
        dxdt = U
        dAxdt = -dFF
        
        L = 0.5 * np.sum(U * U, axis=0, keepdims=True) + 0.5 * np.sum(dFF**2, axis=0, keepdims=True)
        
        return np.vstack((dxdt, dAxdt, -L))

    # --- 4. Solve ---
    bc = make_bc(X0)
    
    try:
        SOL = solve_ivp(
            dynamics, [0., TT], X0,
            method=an_params['ODE_solver'],
            args=(eval_u,),
            rtol=1e-08
        )
        
        if not SOL.success: 
            return None
        
        V_guess, A_guess = bvp_guess(SOL.t.reshape(-1, 1), SOL.y.T)
        X_aug_guess = np.vstack((SOL.y, A_guess.T, V_guess.T))
        
        SOL_bvp = solve_bvp(
            aug_dynamics, bc, SOL.t, X_aug_guess,
            verbose=0,
            tol=an_params['data_tol'],
            max_nodes=an_params['max_nodes']
        )
        
        # CORRECTION 2: Ne pas rejeter si success est False, comme dans la version séquentielle
        # On retourne le résultat tant qu'il existe
        
        V = SOL_bvp.y[-1:] 
        
        return {
            't': SOL_bvp.x.reshape(-1, 1),
            'X': SOL_bvp.y[:dim].T,
            'A': SOL_bvp.y[dim:2*dim].T,
            'V': V.T
        }
        
    except Exception as e:
        # print(f"Exception in worker: {e}")
        return None

def generate_data_parallel(an, V_NN, num_samples, device, n_workers=None, ns=128):
    """
    Parallelized version of generate_data.
    """
    print('Generating data_OC (parallel)...')
    start_time = time.time()
    
    # Default to CPU count if not specified
    if n_workers is None:
        n_workers = min(cpu_count(), num_samples)
        # Cap workers to avoid OOM if models are large
        if n_workers > 16: n_workers = 16 
    
    # 1. Sample Initial Conditions
    x0_int = an.gen_x0(num_samples, Torch=False)
    
    # 2. Prepare State Dicts for Pickling
    # We cannot pickle the whole 'an' object because it contains lambda functions and bound methods
    G_NN_states = []
    for G_NN in an.G_NN_list:
        G_NN.eval()
        # Move to CPU for pickling to avoid CUDA serialization issues in some contexts
        state = {k: v.cpu().clone() for k, v in G_NN.state_dict().items()}
        G_NN_states.append(state)
    
    # Get distribution stats
    x0_initial_np = an.x0_initial.cpu().numpy()
    mu_np = x0_initial_np.mean(axis=0)
    std_np = np.sqrt(x0_initial_np.var(axis=0))
    
    # Params dict
    an_params = {
        'dim': an.dim,
        'TT': an.TT,
        'ODE_solver': an.ODE_solver,
        'data_tol': an.data_tol,
        'max_nodes': an.max_nodes,
        'Round': an.Round,
        'n': an.n,
        'VV': an.VV,
        'G_NN_states': G_NN_states,
        'mu': mu_np,
        'std': std_np,
    }
    
    # V_NN state
    V_NN.eval()
    V_NN_state = {k: v.cpu().clone() for k, v in V_NN.state_dict().items()}
    
    # Device string for worker to recreate device
    device_str = str(device)
    
    # 3. Prepare Args List
    args_list = [
        (x0_int[i, :].copy(), an_params, V_NN_state, ns, device_str, x0_initial_np)
        for i in range(num_samples)
    ]
    
    # 4. Execute Parallel Pool
    # Use 'spawn' to ensure clean CUDA context in workers if using GPU
    ctx = multiprocessing.get_context('spawn')
    
    valid_results = []
    with ctx.Pool(n_workers) as pool:
        # Use imap_unordered for potentially better performance
        for res in pool.imap_unordered(_solve_single_bvp, args_list):
            if res is not None:
                valid_results.append(res)
    
    # 5. Aggregate Results
    dim = an.dim
    if len(valid_results) > 0:
        t_OUT = np.vstack([r['t'] for r in valid_results])
        X_OUT = np.vstack([r['X'] for r in valid_results])
        A_OUT = np.vstack([r['A'] for r in valid_results])
        V_OUT = np.vstack([r['V'] for r in valid_results])
    else:
        t_OUT = np.empty((0, 1))
        X_OUT = np.empty((0, dim))
        A_OUT = np.empty((0, dim))
        V_OUT = np.empty((0, 1))
    
    print(f'Generated {X_OUT.shape[0]} data from {len(valid_results)}/{num_samples} BVP solutions in {time.time() - start_time:.1f} sec\n')
    
    data = {
        't': torch.tensor(t_OUT, dtype=torch.float32, device=device),
        'X': torch.tensor(X_OUT, dtype=torch.float32, device=device),
        'A': torch.tensor(A_OUT, dtype=torch.float32, device=device),
        'V': torch.tensor(V_OUT, dtype=torch.float32, device=device)
    }
    
    return data



from concurrent.futures import ThreadPoolExecutor, as_completed

#================================ solve BVP (ThreadPool) ===========================================#

def _solve_single_bvp_thread(args):
    """
    Worker function to solve a single BVP.
    Thread-safe version using PyTorch networks directly.
    """
    from scipy.interpolate import interp1d

    X0, an, V_NN, device, x0_initial = args

    dim = an.dim
    TT = an.TT
    n = an.n
    VV = an.VV
    G_NN_list = an.G_NN_list

    def eval_u(t_val, x_arr):
        """Evaluate the optimal control u = -nabla V."""
        # Create tensors without tracking gradients initially
        t_tensor = torch.tensor([[t_val]], dtype=torch.float32, device=device)
        x_tensor = torch.tensor(x_arr.reshape(1, -1), dtype=torch.float32, device=device, requires_grad=True)

        # Compute gradient outside of no_grad context
        out = V_NN(t_tensor, x_tensor)
        grad_outputs = torch.ones_like(out)
        grad_inputs = torch.autograd.grad(out, x_tensor, grad_outputs=grad_outputs, create_graph=False)[0]

        u = -grad_inputs.cpu().numpy()
        return u.flatten()

    def bvp_guess(t_arr, x_arr):
        """Generate an initial guess for the BVP."""
        # OPTIMIZATION 2: Vectorized gradient computation
        t_tensor = torch.tensor(t_arr, dtype=torch.float32, device=device)
        x_tensor = torch.tensor(x_arr, dtype=torch.float32, device=device, requires_grad=True)

        V = V_NN(t_tensor, x_tensor)
        grad_outputs = torch.ones_like(V)
        V_x = torch.autograd.grad(V, x_tensor, grad_outputs=grad_outputs, create_graph=False)[0]

        return V.detach().cpu().numpy(), V_x.cpu().numpy()

    def dynamics(t, X, U_fun):
        """State dynamics: dX/dt = u(t, X)."""
        U = U_fun(t, X)
        return U

    def make_bc(X0_in):
        """Create the boundary condition function."""
        def bc(X_aug_0, X_aug_T):
            X0_bc = X_aug_0[:dim]
            AT = X_aug_T[dim:2*dim]
            vT = X_aug_T[2*dim:]
            dFdXT = 0
            return np.concatenate((X0_bc - X0_in, AT - dFdXT, vT))
        return bc

    def aug_dynamics(t, X_aug):
        """
        Augmented dynamics for the BVP system.
        X_aug = [x, A, v] where:
        - x: state (dim,)
        - A: costate (dim,)
        - v: cumulative value (1,)
        """
        x = X_aug[:dim]   # (dim, n_nodes)
        A = X_aug[dim:2*dim]  # (dim, n_nodes)

        # Contrôle optimal: u = -A
        U = -A

        # --- Use the analytic object to get the mean ---
        # an.update_mean handles interpolation and reshaping
        # We need to pass t as tensor or numpy, an.update_mean handles both but returns tensor
        
        # t can be scalar or array. 
        mean_tensor = an.update_mean(t, None)
        mean = mean_tensor.cpu().numpy()
        
        # Ensure mean has correct shape (n_nodes, dim) or (1, dim)
        if mean.ndim == 1:
             mean = mean.reshape(-1, dim)

        # dF/dx = x - mean
        # x est (dim, n_nodes), mean est (n_nodes, dim)
        # Il faut transposer pour aligner les dimensions
        dFF = (x.T - mean).T  # (dim, n_nodes)


        dxdt = U
        dAxdt = -dFF

        # Running cost: L = 0.5 * |u|^2 + F où F = 0.5 * |x - mean|^2
        L = 0.5 * np.sum(U * U, axis=0, keepdims=True) + 0.5 * np.sum(dFF**2, axis=0, keepdims=True)

        return np.vstack((dxdt, dAxdt, -L))
    
    # --- Initial time randomization (optional) ---
    t_start = 0.0 if np.random.rand() < 0.5 else np.random.uniform(0, an.TT * 0.9)

    # --- Solve the BVP ---
    bc = make_bc(X0)

    try:
        # 1. Solve the IVP to get an initial trajectory
        SOL = solve_ivp(
            dynamics, [t_start, TT], X0,
            method=an.ODE_solver,
            args=(eval_u,),
            rtol=1e-08
        )

        if not SOL.success:
            return None

        # 2. Generate initial guess for the BVP
        V_guess, A_guess = bvp_guess(SOL.t.reshape(-1, 1), SOL.y.T)
        X_aug_guess = np.vstack((SOL.y, A_guess.T, V_guess.T))

        # 3. Solve the BVP
        SOL_bvp = solve_bvp(
            aug_dynamics, bc, SOL.t, X_aug_guess,
            verbose=0,
            tol=an.data_tol,
            max_nodes=an.max_nodes
        )

        # Extract results (even if success=False, keep the data)
        V = SOL_bvp.y[-1:]

        return {
            't': SOL_bvp.x.reshape(-1, 1),
            'X': SOL_bvp.y[:dim].T,
            'A': SOL_bvp.y[dim:2*dim].T,
            'V': V.T
        }

    except Exception as e:
        # Print error for debugging (enabled for diagnostics)
        print(f"Exception in BVP solver for X0={X0}: {e}")
        return None


def generate_data_threaded(an, V_NN, num_samples, device, n_workers=None):
    """
    Parallelized version using ThreadPoolExecutor.
    Simpler than multiprocessing since no need to pickle the networks.
    """
    print(f'Generating data_OC (ThreadPool, N={num_samples})...')
    start_time = time.time()
    
    if n_workers is None:
        # Increase default number of workers (scipy releases the GIL)
        n_workers = min(cpu_count(), num_samples, 16)  # Increased from 8 to 16

    print(n_workers, 'workers will be used.\n')
    
    # Set networks to eval mode
    V_NN.eval()
    for G_NN in an.G_NN_list:
        G_NN.eval()
    
    # Get x0_initial for mean field computation
    x0_initial = an.x0_initial
    
    # Sample initial conditions
    x0_samples = an.sample_x0(num_samples, Torch=False)
    
    # Prepare arguments
    args_list = [
        (x0_samples[i, :].copy(), an, V_NN, device, x0_initial) 
        for i in range(num_samples)
    ]
    
    # Execute in parallel with ThreadPoolExecutor
    valid_results = []
    n_success = 0
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_solve_single_bvp_thread, args): i for i, args in enumerate(args_list)}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    valid_results.append(result)
                    n_success += 1
            except Exception as e:
                # Enabled for diagnostics
                print(f"Error processing a future: {e}")
    
    # Aggregate results
    dim = an.dim
    if len(valid_results) > 0:
        t_OUT = np.vstack([r['t'] for r in valid_results])
        X_OUT = np.vstack([r['X'] for r in valid_results])
        A_OUT = np.vstack([r['A'] for r in valid_results])
        V_OUT = np.vstack([r['V'] for r in valid_results])
        # Free valid_results after aggregation
        del valid_results
        gc.collect()
    else:
        t_OUT = np.empty((0, 1))
        X_OUT = np.empty((0, dim))
        A_OUT = np.empty((0, dim))
        V_OUT = np.empty((0, 1))
    
    # Free args_list and x0_samples
    del args_list, x0_samples
    gc.collect()
    
    elapsed = time.time() - start_time
    print(f'Generated {X_OUT.shape[0]} data from {n_success}/{num_samples} BVP solutions in {elapsed:.1f} sec')
    print(f'  → Speed: {num_samples/elapsed:.1f} BVP/sec, {X_OUT.shape[0]/elapsed:.0f} points/sec\n')
    
    data = {
        't': torch.tensor(t_OUT, dtype=torch.float32, device=device),
        'X': torch.tensor(X_OUT, dtype=torch.float32, device=device),
        'A': torch.tensor(A_OUT, dtype=torch.float32, device=device),
        'V': torch.tensor(V_OUT, dtype=torch.float32, device=device)
    }
    
    # Free numpy arrays after conversion to tensors
    del t_OUT, X_OUT, A_OUT, V_OUT
    gc.collect()
    
    return data