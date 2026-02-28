import torch
import time
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
import torch.optim as optim
from src.problem.prb import Analytic

def Solve_HJB(V_NN, num_epoch, t, lr, num_samples, device):
   
  an = Analytic(device)
  V_NN.train()
  optimizer = optim.Adam(V_NN.parameters(), lr)
  T = an.TT*torch.ones(num_samples,1, device=device)
  old_loss = 1
  loss = []  

  for epoch in range(num_epoch+1):

      t_rand = torch.rand(num_samples, 1, device=device, requires_grad=True) * an.TT
      x_rand = an.sample_x0(num_samples).requires_grad_(True)

      V_nn = V_NN(t_rand, x_rand)
      V_nn_t = torch.autograd.grad(V_nn, t_rand, torch.ones_like(V_nn), create_graph=True)[0]
      V_nn_x = torch.autograd.grad(V_nn, x_rand, torch.ones_like(V_nn), create_graph=True)[0]

      Loss_hjb = torch.mean((V_nn_t + an.ham(t_rand, x_rand, V_nn_x))**2)
      Loss_bc = torch.mean((V_NN(T, x_rand) - an.psi_func(x_rand))**2)


      optimizer.zero_grad()


      Loss_hjb.backward()


      optimizer.step()


      if epoch % 1000 == 0:
        loss.append(old_loss)    
        new_loss = Loss_hjb.item()
        print(f"Iteration {epoch}: Loss = {Loss_hjb.item():.4e}")
        if new_loss>min(loss):
            x_rand = x_rand
        else:   
            x_rand = an.sample_x0(num_samples).requires_grad_(True)
        old_loss = new_loss    

  return V_NN

# def Approximate_v(V_NN, data, num_epoch, t, lr, num_samples, device):
   
#     an = Analytic(device)
#     V_NN.train()
#     optimizer = optim.Adam(V_NN.parameters(), lr)
#     T = an.TT * torch.ones(num_samples, 1, device=device)
#     old_loss = 1.0
#     loss_hist = []

#     # Préparer les données supervisées pour le calcul de gradient (requires_grad=True)
#     # On clone pour ne pas modifier les tenseurs originaux du dictionnaire
#     t_data = data['t'].clone().detach().requires_grad_(True)
#     x_data = data['X'].clone().detach().requires_grad_(True)
#     v_target = data['V'].detach()
#     a_target = data['A'].detach()

#     for epoch in range(num_epoch + 1):
#         # --- 1. Partie HJB (Points aléatoires) ---
#         t_rand = torch.rand(num_samples, 1, device=device, requires_grad=True) * an.TT
#         x_rand = an.sample_x0(num_samples).requires_grad_(True)

#         V_nn = V_NN(t_rand, x_rand)
#         V_nn_t = torch.autograd.grad(V_nn, t_rand, torch.ones_like(V_nn), create_graph=True)[0]
#         V_nn_x = torch.autograd.grad(V_nn, x_rand, torch.ones_like(V_nn), create_graph=True)[0]

#         Loss_hjb = torch.mean((V_nn_t + an.ham(t_rand, x_rand, V_nn_x))**2)

#         # --- 2. Partie Supervisée (Données PMP) ---
#         # On ré-évalue V_NN sur les données générées pour avoir le graphe de calcul
#         V_pred = V_NN(t_data, x_data)
        
#         # Calcul du gradient prédit (nécessite create_graph=True pour backpropager sur les poids)
#         V_pred_x = torch.autograd.grad(V_pred, x_data, torch.ones_like(V_pred), create_graph=True)[0]

#         Loss_v = torch.mean((V_pred - v_target)**2)
#         Loss_v_x = torch.mean((V_pred_x - a_target)**2)

#         # Somme des pertes
#         Loss_total = 0*Loss_hjb + Loss_v + Loss_v_x

#         optimizer.zero_grad()
#         Loss_total.backward()
#         optimizer.step()

#         if epoch % 1000 == 0:
#             loss_hist.append(old_loss)
#             new_loss = Loss_total.item()
#             print(f"Iteration {epoch}: Loss_V={Loss_v.item():.3e}, "
#                   f"Loss_V_x={Loss_v_x.item():.3e}, Loss_HJB={Loss_hjb.item():.3e}, "
#                   f"Loss_total={Loss_total.item():.3e}")
#             old_loss = new_loss

#     return V_NN

def Approximate_v(V_NN, data, num_epoch, t, lr, num_samples, device):
   
  an = Analytic(device)
  V_NN.train()
  optimizer = optim.Adam(V_NN.parameters(), lr)
  x_rand = an.sample_x0(num_samples).requires_grad_(True)
  T = an.TT*torch.ones(num_samples,1, device=device)
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
    
      V_pred = V_NN(data['t'], data['X'])
    
      # Calcul du gradient prédit (nécessite create_graph=True pour backpropager sur les poids)
      V_pred_x = torch.autograd.grad(V_pred, data['X'], torch.ones_like(V_pred), create_graph=True)[0]

      Loss_hjb = torch.mean(( V_nn_t + an.ham(t,x_rand,V_nn_x))**2) +  torch.mean((V_NN(T, x_rand)-an.psi_func(x_rand))**2)
      Loss_v = torch.mean((V_pred - data['V'])**2)
      Loss_v_x = torch.mean((V_pred_x - data['A'])**2)                   

      Loss_total =   0*Loss_hjb + Loss_v + Loss_v_x
    
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

  return V_NN



def generate_data(V_NN, num_samples, t, lr, num_samples_hjb, device, num_epoch=100):
    
    an = Analytic(device)
    
    def eval_u(t, x):
            t_tensor = torch.tensor([[t]], dtype=torch.float32, device=device)
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device).view(1, -1)

            p = V_NN.get_grad(t_tensor, x_tensor).detach().cpu().numpy().T  # (dim,1)
            norm_p = np.linalg.norm(p, axis=0, keepdims=True)

            if norm_p < 2.0:
                u = -0.5 * p
            elif norm_p > 2.0:
                u = -p / norm_p
            else:
                u = np.zeros_like(p)

            return u.flatten()
    
    def bvp_guess(t, x):
        V_NN.eval()
        V = V_NN(torch.tensor(t, dtype=torch.float32, device=device), 
                 torch.tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy()
        V_x = V_NN.get_grad(t, x).detach().cpu().numpy()
        return V, V_x

    print('Generating data...')

    dim = an.dim

    X_OUT = np.empty((0, dim))
    A_OUT = np.empty((0, dim))
    V_OUT = np.empty((0, 1))
    t_OUT = np.empty((0, 1))

    Ns_sol = 0
    failure_count = 0
    start_time = time.time()
    
    x0_int = an.gen_x0(num_samples)

    # ----------------------------------------------------------------------

    while Ns_sol < num_samples:
        
        print('Solving BVP #', Ns_sol+1, '...', end='\r')
        
        max_failures = num_samples - Ns_sol

        #X0 = an.gen_x0(1)
        X0 = x0_int[Ns_sol,:]
        
        bc = an.make_bc(X0)

        # Integrates the closed-loop system (NN controller)

        SOL = solve_ivp(an.dynamics, [0., an.TT], X0,
                        method=an.ODE_solver,
                        args=(eval_u,),
                        rtol=1e-08)


        V_guess, A_guess = bvp_guess(SOL.t.reshape(1,-1).T, SOL.y.T)

        try:
            # Solves the two-point boundary value problem

            X_aug_guess = np.vstack((SOL.y, A_guess.T, V_guess.T))
            SOL = solve_bvp(an.aug_dynamics, bc, SOL.t, X_aug_guess,
                            verbose=0,
                            tol=an.data_tol,
                            max_nodes=an.max_nodes)
            # Save only successful solutions
            if SOL.success:
                V = SOL.y[-1:] + an.terminal_cost(SOL.y[:dim, -1])
                t_OUT = np.vstack((t_OUT, SOL.x.reshape(1, -1).T))
                X_OUT = np.vstack((X_OUT, SOL.y[:dim].T))
                A_OUT = np.vstack((A_OUT, SOL.y[dim:2*dim].T))
                V_OUT = np.vstack((V_OUT, V.T))
                Ns_sol += 1
                failure_count = 0  # Reset failure count on success
            else:
                print(f"Solver failed ({failure_count + 1}/{max_failures}): {SOL.message}")
                failure_count += 1
                if Ns_sol < 128:
                    V_NN = Solve_HJB(V_NN, num_epoch, t, lr, num_samples_hjb, device)
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
            # 'U': an.U_star(torch.vstack((torch.tensor(X_OUT, dtype=torch.float32, device=device), 
            #                                        torch.tensor(A_OUT, dtype=torch.float32, device=device))))}

    return data
