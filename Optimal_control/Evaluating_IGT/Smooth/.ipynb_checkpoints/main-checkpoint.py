import multiprocessing
import numpy as np
import argparse
import time
from model import *
from src.solve import *
import sys
from src.problem.prb import Analytic

# ==========================
#      Logging Setup
# ==========================
class Tee(object):
    """Class to write output to both a file and the Jupyter Notebook in real-time."""
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout  # Save original stdout

    def write(self, data):
        self.stdout.write(data)  # Print to Jupyter
        self.file.write(data)  # Write to file
        self.file.flush()  # Ensure immediate writing

    def flush(self):
        self.stdout.flush()
        self.file.flush()

# Redirect stdout to both Jupyter and a log file
sys.stdout = Tee("training_output.log")

parser = argparse.ArgumentParser()
parser.add_argument('--t0', type=int, default=0)
parser.add_argument('--t_final', type=int, default=1)
parser.add_argument('--dt', type=int, default=0.05)
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--num_samples_hjb', type=int, default=5000)
parser.add_argument('--num_samples_bvp', type=int, default=264)
parser.add_argument('--num_points_test', type=int, default=1000)
parser.add_argument('--Max_Round', type=float, default=5)
parser.add_argument('--num_epoch', type=float, default=11000)
parser.add_argument('--num_itr', type=float, default=30000)
parser.add_argument('--freq', type=float, default=1000)
parser.add_argument('--ns', default=128, help='Network size')
parser.add_argument('--lr', default=1e-4)
parser.add_argument('--lrr', default=1e-4)
parser.add_argument('--betas', default=(0.5, 0.9), help='Adam only')
parser.add_argument('--weight_decay',default=1e-3)
parser.add_argument('--act_func',default= lambda x: torch.tanh(x), help='Activation function for discriminator')
parser.add_argument('--hh', default=0.5, help='ResNet step-size')
args = parser.parse_args()

torch_seed = np.random.randint(low=-sys.maxsize - 1, high=sys.maxsize)
torch.random.manual_seed(torch_seed)
np_seed = np.random.randint(low=0, high=2 ** 32 - 1)
np.random.seed(np_seed)

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

def relative_error(u_exct, u_predicted):

    norm_u_exct = torch.norm(u_exct)  # Compute the norm of u_exct
    error = torch.norm(u_exct - u_predicted)  # Compute the error ||u_exct - u_predicted||
    relative_error = error / norm_u_exct  # Compute the relative error
    
    return relative_error

def test(V_NN, num_points, dim):
    x = 2*torch.rand(num_points, dim, device=device)-1
    #x = create_meshgrid_2d(num_points)
    #x = torch.linspace(-1, 1, num_points, device=device).unsqueeze(1)
    with torch.no_grad():
        V_NN.eval()
        # t = 0
        t = 0*torch.ones(num_points,1, device=device)
        v_pred = V_NN(t,x)
        v_exact = an.V_exact(x,t=0)
        err = relative_error(v_exact, v_pred)
        print("Relative_Error_t0 = ", err)

        # t = 0.5
        t = 0.5*torch.ones(num_points,1, device=device)
        v_pred = V_NN(t,x)
        v_exact = an.V_exact(x,t=0.5)
        err = relative_error(v_exact, v_pred)
        print("Relative_Error_t0.5 = ", err)

        # t = 1
        t = torch.ones(num_points,1, device=device)
        v_pred = V_NN(t,x)
        v_exact = an.V_exact(x,t=1.0)
        err = relative_error(v_exact, v_pred)
        print("Relative_Error_tf = ", err)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device == torch.device('cpu'):
    print('NOTE: USING ONLY THE CPU')

an = Analytic(device)    

if __name__ == "__main__":
    
    #act_funcs = lambda x: torch.tanh(x)
    start_time = time.time()
    V_NN = V_Net(dim=args.dim, ns=args.ns, act_func=args.act_func, hh=args.hh,
                            device=device, psi_func=an.psi_func, TT=an.TT).to(device)
    
    
    t = torch.linspace(args.t0, args.t_final, args.num_samples_hjb, device=device).unsqueeze(1)
    
    print('Approximate solution by HJB \n')
    

    V_NN = Solve_HJB(V_NN, args.num_epoch, t, args.lr, args.num_samples_hjb, device)
    
    #test(V_NN, args.num_points_test, args.dim)
    
    for Round in range(args.Max_Round):
        
        print('Round: ', Round, '\n')
        
        data = generate_data(V_NN, args.num_samples_bvp, device)
        print('Approximate solution of BVP \n')
        V_NN = Approximate_v(V_NN, data, args.num_itr, t, args.lrr, args.num_samples_hjb, Round, device)
        #test(V_NN, args.num_points_test, args.dim)
    print('Time = %.1f' % (time.time() - start_time), 'sec \n')    
    torch.save(V_NN.state_dict(), 'V_NN_d100.pth')
    
    # --- Restore stdout after logging ---
#     sys.stdout.file.close()
#     sys.stdout = sys.__stdout__
    
    
