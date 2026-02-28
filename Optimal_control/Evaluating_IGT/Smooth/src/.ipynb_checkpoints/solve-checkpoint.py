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

      Loss = torch.mean(( V_nn_t + an.ham(t,x_rand,V_nn_x))**2) #+  torch.mean((V_NN(T, x_rand)-an.psi_func(x_rand))**2)


      optimizer.zero_grad()


      Loss.backward()


      optimizer.step()


      if epoch % 1000 == 0:
        loss.append(old_loss)    
        new_loss = Loss.item()
        print(f"Iteration {epoch}: Loss = {Loss.item():.4e}")
        if new_loss>min(loss):
            x_rand = x_rand
        else:   
            x_rand = an.sample_x0(num_samples).requires_grad_(True)
        old_loss = new_loss    

  return V_NN

def Approximate_v(V_NN, data, num_epoch, t, lr, num_samples, Round, device):
   
    an = Analytic(device)
    V_NN.train()
    optimizer = optim.Adam(V_NN.parameters(), lr)
    x_rand = an.sample_x0(num_samples).requires_grad_(True)
    T = an.TT*torch.ones(num_samples,1, device=device)
    old_loss = 1
    loss = []
    los_hjb = []
    los_v = []
    los_v_x = []  
    
#     if Round == 0:
#         tol = 1e-7
#     elif Round == 1:
#         tol = 5e-8
#     elif Round == 2:
#         tol = 25e-9
# #     elif Round == 3:
# #         tol = 15e-5
# #     elif Round == 4:
# #         tol = 1e-4




    for epoch in range(num_epoch+1):

        t = t.requires_grad_(True)

        V_nn =  V_NN(t, x_rand)

        V_nn_t = torch.autograd.grad(outputs=V_nn, inputs=t,
                                          grad_outputs=torch.ones_like(V_nn),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]

        V_nn_x = torch.autograd.grad(outputs=V_nn, inputs=x_rand,
                                          grad_outputs=torch.ones_like(V_nn),
                                          create_graph=True, retain_graph=True, only_inputs=True)[0]

        Loss_hjb = torch.mean(( V_nn_t + an.ham(t,x_rand,V_nn_x))**2) #+  torch.mean((V_NN(T, x_rand)-an.psi_func(x_rand))**2)
        Loss_v = torch.mean((V_NN(data['t'], data['X']) - data['V'])**2)
        Loss_v_x = torch.mean((V_NN.get_grad(data['t'], data['X']) - data['A'])**2)

        los_hjb.append(Loss_hjb.item())
        los_v.append(Loss_v.item())
        los_v_x.append(Loss_v_x.item())

        Loss_total =   0.01*Loss_hjb + Loss_v + Loss_v_x

        optimizer.zero_grad()


        Loss_total.backward()


        optimizer.step()
        
#         if Loss_total<tol and epoch>15000:
#             print(f"Iteration {epoch}: Loss_V = {Loss_v.item():.4e}, Loss_V_x = {Loss_v_x.item():.4e}, Loss_HJB = {Loss_hjb.item():.4e}, Loss_total = {Loss_total.item():.4e}")
#             break


        if epoch % 1000 == 0:
            loss.append(old_loss)    
            new_loss = Loss_total.item()
            print(f"Iteration {epoch}: Loss_V = {Loss_v.item():.4e}, Loss_V_x = {Loss_v_x.item():.4e}, Loss_HJB = {Loss_hjb.item():.4e}, Loss_total = {Loss_total.item():.4e}")
            if new_loss>min(loss):
               x_rand = x_rand
            else:   
               x_rand = an.sample_x0(num_samples).requires_grad_(True)
            old_loss = new_loss    
        
    #np.savez(f'Loss_Round_{Round}.npz', los_hjb=los_hjb, los_v=los_v, los_v_x=los_v_x)
    return V_NN


def generate_data(V_NN, num_samples, device):
    
    an = Analytic(device)
    
    def eval_u(t, x):
        u = -0.5 * V_NN.get_grad(t, x).detach().cpu().numpy()
        return u
    
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
    start_time = time.time()

    # ----------------------------------------------------------------------

    while Ns_sol < num_samples:
        
        print('Solving BVP #', Ns_sol+1, '...', end='\r')

        X0 = an.gen_x0(1)
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
            # 'U': an.U_star(torch.vstack((torch.tensor(X_OUT, dtype=torch.float32, device=device), 
            #                                        torch.tensor(A_OUT, dtype=torch.float32, device=device))))}


    return data
   

# def generate_data(V_NN, num_samples, device):
    
#     an = Analytic(device)
    
#     def eval_u(t, x):
#         u = -0.5 * V_NN.get_grad(t, x).detach().cpu().numpy()
#         return u
    
#     def bvp_guess(t, x):
#         V_NN.eval()
#         V = V_NN(torch.tensor(t, dtype=torch.float32, device=device), 
#                  torch.tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy()
#         V_x = V_NN.get_grad(t, x).detach().cpu().numpy()
#         return V, V_x

#     print('Generating data...')

#     dim = an.dim

#     X_OUT = np.empty((0, dim))
#     A_OUT = np.empty((0, dim))
#     V_OUT = np.empty((0, 1))
#     t_OUT = np.empty((0, 1))
#     X0_pool = an.gen_x0(num_samples)
#     Ns_sol = 0
#     start_time = time.time()

#     # ----------------------------------------------------------------------

#     while Ns_sol < num_samples:
        
#         print('Solving BVP #', Ns_sol+1, '...', end='\r')

#         X0 = X0_pool[:,Ns_sol]
#         t0 = np.array([[0.]])
#         bc = an.make_bc(X0)
       
#         V_guess0, A_guess0 = bvp_guess(t0, X0.reshape(-1,1))

#         try:
#             t_guess = np.array([0.])
#             X_guess = np.vstack((X0.reshape(-1,1), A_guess0.T, V_guess0.T))
#             tol = 1e-01
#             ##### Time-marching to build from t0 to tf #####
#             for k in range(an.tseq.shape[0]):
#                 if tol >= 2.*an.data_tol:
#                     tol /= 2.
#                 if k == an.tseq.shape[0] - 1:
#                     tol = an.data_tol
#                 t_guess = np.concatenate((t_guess, an.tseq[k:k+1]))
#                 X_guess = np.hstack((X_guess, X_guess[:,-1:]))
                
#                 SOL = solve_bvp(an.aug_dynamics, bc, t_guess, X_guess,
#                                 verbose=0, tol=tol, max_nodes=an.max_nodes)

#                 if not SOL.success:
#                     print(SOL.message)
#                     warnings.warn(Warning())
#                 t_guess = SOL.x
#                 X_guess = SOL.y


#             Ns_sol += 1
#             V = SOL.y[-1:] + an.terminal_cost(SOL.y[:dim,-1])

#             t_OUT = np.vstack((t_OUT, SOL.x.reshape(1,-1).T))
#             X_OUT = np.vstack((X_OUT, SOL.y[:dim].T))
#             A_OUT = np.vstack((A_OUT, SOL.y[dim:2*dim].T))
#             V_OUT = np.vstack((V_OUT, V.T))

#         except Warning:
#             pass

#     print('Generated', X_OUT.shape[0], 'data from', Ns_sol,
#         'BVP solutions in %.1f' % (time.time() - start_time), 'sec \n')

#     data = {'t': torch.tensor(t_OUT, dtype=torch.float32, device=device), 
#             'X': torch.tensor(X_OUT, dtype=torch.float32, device=device), 
#             'A': torch.tensor(A_OUT, dtype=torch.float32, device=device), 
#             'V': torch.tensor(V_OUT, dtype=torch.float32, device=device)}
#             # 'U': an.U_star(torch.vstack((torch.tensor(X_OUT, dtype=torch.float32, device=device), 
#             #                                        torch.tensor(A_OUT, dtype=torch.float32, device=device))))}

#     return data    