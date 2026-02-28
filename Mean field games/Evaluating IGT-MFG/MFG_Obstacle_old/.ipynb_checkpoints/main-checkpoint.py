import sys
import time
import torch
import numpy as np
import argparse
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

from model import *
from src.solve import *
from src.problem.prb import Obstacle

# ==========================
#      Logging Setup
# ==========================
class Tee(object):
    """Class to write output to both a file and the terminal in real-time."""
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

# Redirect stdout to both console and log file
sys.stdout = Tee("training_output.log")

# ==========================
#      Argument Parsing
# ==========================
parser = argparse.ArgumentParser()
parser.add_argument('--t0', type=int, default=0, help='Initial time')
parser.add_argument('--t_final', type=float, default=1.0, help='Final time')
parser.add_argument('--N', type=int, default=20, help='Number of time steps')
parser.add_argument('--dim', type=int, default=2, help='Dimension')
parser.add_argument('--num_samples_hjb', type=int, default=264)
parser.add_argument('--num_samples_bvp', type=int, default=32)
parser.add_argument('--num_samples_gen', type=int, default=128)
parser.add_argument('--num_samples_conv', type=int, default=264)
parser.add_argument('--num_points_test', type=int, default=500)
parser.add_argument('--Max_Round', type=int, default=5)
parser.add_argument('--num_epoch_hjb', type=int, default=5000)
parser.add_argument('--num_epoch_v', type=int, default=5000)
parser.add_argument('--num_epoch_gen', type=int, default=5000)
parser.add_argument('--freq', type=int, default=1000)
parser.add_argument('--ns_v', default=128)
parser.add_argument('--ns_g', default=128)
parser.add_argument('--lr_v', default=1e-4)
parser.add_argument('--lr_v2', default=1e-4)
parser.add_argument('--lr_g', default=1e-3)
parser.add_argument('--betas', default=(0.5, 0.9))
parser.add_argument('--weight_decay', default=1e-3)
parser.add_argument('--act_func_v', default=lambda x: torch.tanh(x))
parser.add_argument('--act_func_g', default=lambda x: torch.relu(x))
parser.add_argument('--hh', default=0.5)
args = parser.parse_args()

# ==========================
#      Set Random Seeds
# ==========================
torch_seed = np.random.randint(low=-sys.maxsize - 1, high=sys.maxsize)
torch.random.manual_seed(torch_seed)
np_seed = np.random.randint(low=0, high=2**32 - 1)
np.random.seed(np_seed)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device == torch.device('cpu'):
    print('NOTE: USING ONLY THE CPU')

# ==========================
#     Wasserstein Helper
# ==========================
def W_gem(obs, G_history, G_k, x0_ini, m0, t_grid):
    W_k = wasserstein_fp(obs, G_history, G_k, x0_ini, m0, t_grid, p=1, blur=0.5, device=device)
    print(f"k {obs.n:2d} │ L1={np.linalg.norm(W_k,1):.3e} │ "
          f"L2={np.linalg.norm(W_k,2):.3e} │ L∞={W_k.max():.3e}")
    return W_k.max()

# ==========================
#         Load Data
# ==========================
data = np.load('data1.npz') 
x0_ini = torch.tensor(data['x'], dtype=torch.float32, device=device)
m0 = x0_ini
mu = x0_ini.mean(axis=0)
std = torch.sqrt(x0_ini.var(axis=0))

# ==========================
#     Main Training Loop
# ==========================
def main():
    obs = Obstacle(G_NN_list=[], Round=0, n=0, x0_initial=x0_ini, device=device, VV=1)

    V_NN = V_Net(args.dim, args.ns_v, args.act_func_v, args.hh,
                 device=device, psi_func=obs.psi_func, TT=args.t_final).to(device)
    
    V_NN2 = V_Net(args.dim, args.ns_v, args.act_func_v, args.hh,
                  device=device, psi_func=obs.psi_func, TT=args.t_final).to(device)
    
    G_NN = G_Net(args.dim, args.ns_g, args.act_func_g, args.hh,
                 device=device, mu=mu, std=std, TT=args.t_final).to(device)

    t = torch.linspace(args.t0, args.t_final, args.num_samples_hjb, device=device).unsqueeze(1)
    t_g = torch.linspace(args.t0, args.t_final, args.num_samples_gen, device=device).unsqueeze(1)
    tt = torch.linspace(args.t0, args.t_final, 21, device=device).unsqueeze(1)

    tol = 1e-3
    Delta = 1 + tol

    for Round in range(args.Max_Round):
        if Delta <= tol:
            break

        print(f"\n========= Round {Round} =========\n")
        delta = 1 + tol
        nmax = 15
        G_NN_list = []

        if Round != 0:
            G_NN_list.append(copy.deepcopy(G_NN_new))

        for n in range(nmax):
            if delta <= tol:
                break

            print(f"\n>>> Computing Best Response {n}...\n")
            start_time = time.time()

            # Update Obstacle object
            obs = Obstacle(G_NN_list, Round, n, x0_ini, device, VV=1)

            # Step 1: Train V_NN
            V_NN = train_v_nn(obs, V_NN, args.num_epoch_v, args.num_epoch_hjb, t, 
                              args.lr_v, args.num_samples_hjb, args.num_samples_bvp, Round, device, VV=1)

            # Step 2: Simulate trajectories
            t_tr, X_tr, t_OUT, X_OUT, x_out = sim_points(obs, V_NN, args.num_samples_gen,
                                                         args.N, args.t0, args.t_final, device)

            # Step 3: Train Generator
            G_NN, G_NN_new = Train_Gen(obs, G_NN, V_NN, t_tr, X_tr, X_OUT, args.num_epoch_gen,
                                       t_g, args.lr_g, args.num_samples_gen, Round, device)

            # Step 4: Evaluate convergence (Wasserstein)
            delta = W_gem(obs, G_NN_list, G_NN_new, x0_ini, m0, tt)

            # Save models
            G_NN_list.append(copy.deepcopy(G_NN_new))
            torch.save(G_NN_new.state_dict(), f"G_NNN_round{Round}_n{n}.pth")
            torch.save(V_NN.state_dict(), f"V_NNN_round{Round}_n{n}.pth")

            # Step 5: Compute J
            J = Comp_J(obs, tt, x0_ini, V_NN)

            # Step 6: Train second value network
            obs = Obstacle(G_NN_list, Round, n, x0_ini, device, VV=2)
            V_NN = train_v_nn(obs, V_NN, args.num_epoch_v, args.num_epoch_hjb, t, 
                              args.lr_v2, args.num_samples_hjb, args.num_samples_bvp, Round, device, VV=2)

            # Step 7: Evaluate exploitability
            V0 = Comp_V(x0_ini, V_NN)
            exploi = abs(V0 - J)
            print(f"\033[1;32mExploitability: {exploi:.6f}\033[0m")
            print(f"Step Time: {time.time() - start_time:.1f} sec\n")

            # Update generator for next iteration
            G_NN = copy.deepcopy(G_NN_new)

        Delta = delta


if __name__ == "__main__":
    main()

