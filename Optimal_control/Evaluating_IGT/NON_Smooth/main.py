import multiprocessing
import numpy as np
import argparse
import time
from model import *
from src.solve import *
import sys
from src.problem.prb import Analytic


parser = argparse.ArgumentParser()
parser.add_argument('--t0', type=int, default=0)
parser.add_argument('--t_final', type=int, default=1)
parser.add_argument('--dt', type=int, default=0.05)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--num_samples_hjb', type=int, default=5000)
parser.add_argument('--num_samples_bvp', type=int, default=256)
parser.add_argument('--Max_Round', type=float, default=5)
parser.add_argument('--num_epoch', type=float, default=10000)
parser.add_argument('--num_itr', type=float, default=30000)
parser.add_argument('--freq', type=float, default=1000)
parser.add_argument('--ns', default=128, help='Network size')
parser.add_argument('--lr', default=1e-3)
parser.add_argument('--lrr', default=1e-4)
parser.add_argument('--betas', default=(0.5, 0.9), help='Adam only')
parser.add_argument('--weight_decay',default=1e-3)
parser.add_argument('--act_func',default=torch.nn.Tanh(), help='Activation function for discriminator')
parser.add_argument('--hh', default=0.5, help='ResNet step-size')
args = parser.parse_args()

torch_seed = np.random.randint(low=-sys.maxsize - 1, high=sys.maxsize)
torch.random.manual_seed(torch_seed)
np_seed = np.random.randint(low=0, high=2 ** 32 - 1)
np.random.seed(np_seed)


def plot_v(x, v_exact, v_pred, t_value):
    x_np = x.cpu().numpy().flatten()
    v_ex_np = v_exact.cpu().numpy().flatten()
    v_pr_np = v_pred.cpu().numpy().flatten()
    err_np  = abs(v_pr_np - v_ex_np)

    plt.figure(figsize=(8,5))
    plt.plot(x_np, v_ex_np, label="Exact V", linewidth=2)
    plt.plot(x_np, v_pr_np, "--", label="Predicted V", linewidth=2)
    plt.title(f"Value Function at t = {t_value}")
    plt.xlabel("x")
    plt.ylabel("V(t,x)")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(x_np, err_np, color="red")
    plt.title(f"Absolute Error |V_pred - V_exact| at t = {t_value}")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.grid()
    plt.show()



def relative_error(u_exct, u_predicted):

    norm_u_exct = torch.norm(u_exct)  # Compute the norm of u_exct
    error = torch.norm(u_exct - u_predicted)  # Compute the error ||u_exct - u_predicted||
    relative_error = error / norm_u_exct  # Compute the relative error
    
    return relative_error

def create_meshgrid_2d(num_points):
    # Create 1D tensors for x and y coordinates
    x = torch.linspace(-1, 1, num_points, device=device)
    y = torch.linspace(-1, 1, num_points, device=device)

    # Create a meshgrid using torch.meshgrid()
    X, Y = torch.meshgrid(x, y)

    # Flatten X and Y and concatenate them
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    points = torch.stack((X_flat, Y_flat), dim=1)

    return points


def test(V_NN):
    # Generate 2D grid
    #x = create_meshgrid_2d(100) # Shape (10000, 2)
    # Generate random points in [-1, 1]^d
    x = 2 * torch.rand(100000, 100, device=device) - 1
    V_NN.eval()

    # List of time values to test
    times = [0.0, 0.25, 0.5, 0.75, 1.0]

    with torch.no_grad():
        for t_val in times:
            # Create time tensor (N, 1)
            t_tensor = torch.full((x.shape[0], 1), t_val, device=device)
            
            # Prediction (N, 1)
            v_pred = V_NN(t_tensor, x)
            
            # Exact solution
            # IMPORTANT: Ensure that v_exact has the same shape as v_pred
            v_exact = an.V_exact(x, t=t_val).view_as(v_pred)
            
            # Compute error
            err = relative_error(v_exact, v_pred)
            
            print(f"Relative_Error_t={t_val} = {err.item():.5e}")
            
            # Optional: Plot if 1D or 2D slice
            # if args.dim == 1: plot_v(...)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device == torch.device('cpu'):
    print('NOTE: USING ONLY THE CPU')

an = Analytic(device)    

if __name__ == "__main__":
    
    
    V_NN = V_Net(dim=args.dim, ns=args.ns, act_func=args.act_func, hh=args.hh,
                            device=device, psi_func=an.psi_func, TT=an.TT).to(device)
    
    
    t = torch.linspace(args.t0, args.t_final, args.num_samples_hjb, device=device).unsqueeze(1)
    
    print('Approximate solution by HJB \n')
    start_time = time.time()

    V_NN = Solve_HJB(V_NN, args.num_epoch, t, args.lr, args.num_samples_hjb, device)
    print('HJb_Solved in %.1f' % (time.time() - start_time), 'sec \n')
    test(V_NN)
    
    for Round in range(args.Max_Round):
        
        print('Round: ', Round, '\n')
        data = generate_data(V_NN, args.num_samples_bvp, t, args.lr, args.num_samples_hjb, device, num_epoch=100)
        print('Approximate solution of BVP \n')
        V_NN = Approximate_v(V_NN, data, args.num_itr, t, args.lrr, args.num_samples_hjb, device)
        test(V_NN)
    torch.save(V_NN.state_dict(), 'V_NN100_.pth')

