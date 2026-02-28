import multiprocessing
import numpy as np
import argparse
import time
import gc
from model import *
from src.solve import *
import sys
from src.problem.prb import Analytic

def clear_gpu_memory():
    """Clear GPU and RAM memory."""
    # Double gc.collect call to release cyclic references
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def save_G_NN_to_disk(G_NN, round_idx, n_idx, folder="G_NN_cache_50"):
    """Save a G_NN model to disk and return the file path."""
    import os
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"G_NN_r{round_idx}_n{n_idx}.pth")
    torch.save(G_NN.state_dict(), path)
    return path

def load_G_NN_from_disk(path, dim, ns, act_func, hh, device, mu, std, TT):
    """Load a G_NN model from disk."""
    G_NN = G_Net(dim=dim, ns=ns, act_func=act_func, hh=hh,
                 device=device, mu=mu, std=std, TT=TT).to(device)
    G_NN.load_state_dict(torch.load(path, map_location=device))
    G_NN.eval()
    return G_NN

def get_G_NN_list_from_paths(paths, dim, ns, act_func, hh, device, mu, std, TT):
    """Load a list of G_NN models from their saved paths on disk."""
    G_NN_list = []
    for path in paths:
        G_NN = load_G_NN_from_disk(path, dim, ns, act_func, hh, device, mu, std, TT)
        G_NN_list.append(G_NN)
    return G_NN_list


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
sys.stdout = Tee("training_output50.log")


parser = argparse.ArgumentParser()
parser.add_argument('--t0', type=int, default=0, help='Initial time')
parser.add_argument('--t_final', type=int, default=1, help='Final time')
parser.add_argument('--N', type=int, default=20, help='Number of time steps')
parser.add_argument('--dim', type=int, default=50, help='Dimension')
parser.add_argument('--num_samples_hjb', type=int, default=512, help='Number of samples for training HJB')
parser.add_argument('--num_samples_mfg', type=int, default=512, help='Number of samples for training MFG')
parser.add_argument('--num_samples_bvp', type=int, default=512, help='Number of samples for BVP')
parser.add_argument('--num_samples_gen', type=int, default=512, help='Number of samples for training Generator')
parser.add_argument('--num_samples_conv', type=int, default=512, help='Number of samples for computing convolution')
parser.add_argument('--num_points_test', type=int, default=100000)
parser.add_argument('--Max_Round', type=float, default=1, help='Number of rounds')
parser.add_argument('--num_epoch_hjb', type=float, default=1000, help='Number of training iterations for HJB')
parser.add_argument('--num_epoch_mfg', type=float, default=3000, help='Number of training iterations for MFG')
parser.add_argument('--num_epoch_v', type=float, default=5000, help='Number of training iterations for V')
parser.add_argument('--num_epoch_gen', type=float, default=3000, help='Number of training iterations for Generator')
parser.add_argument('--freq', type=float, default=1000)
parser.add_argument('--ns_v', default=128, help='Network size of V_net')
parser.add_argument('--ns_g', default=128, help='Network size of G_net')
parser.add_argument('--lr', default=1e-3, help='Learning rate of MFG')
parser.add_argument('--lr_v', default=1e-4, help='Learning rate of V')
parser.add_argument('--lr_v2', default=1e-4, help='Learning rate of V2')
parser.add_argument('--lr_g', default=1e-4, help='Learning rate of G')
parser.add_argument('--betas', default=(0.5, 0.9), help='Adam only')
parser.add_argument('--weight_decay', default=0*1e-3)
parser.add_argument('--act_func_v', default=lambda x: torch.tanh(x), help='Activation function for V')
parser.add_argument('--act_func_g', default=lambda x: torch.relu(x), help='Activation function for G')
parser.add_argument('--hh', default=0.5, help='ResNet step-size')
args = parser.parse_args()


torch_seed = np.random.randint(low=-sys.maxsize - 1, high=sys.maxsize)
torch.random.manual_seed(torch_seed)
np_seed = np.random.randint(low=0, high=2 ** 32 - 1)
np.random.seed(np_seed)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device == torch.device('cpu'):
    print('NOTE: USING ONLY THE CPU')

def psi_func(xx_inp):
    """
    The final-time cost function.
    """
    return torch.zeros(xx_inp.size(0), device=device)

def V_exact(x, T, mu, dim, t):
    """Compute the exact value function for verification."""
    Pi_t = (np.exp(2*T - t) - np.exp(t)) / (np.exp(2*T - t) + np.exp(t))
    s_t = -Pi_t * mu
    c_t = dim/2 * Pi_t * mu**2

    out = 0.5 * Pi_t * torch.sum(x**2, dim=1, keepdim=True) + s_t * torch.sum(x, dim=1, keepdim=True) + c_t

    return out

def relative_error(u_exact, u_predicted):
    """Compute the relative L2 error."""
    norm_exact = torch.norm(u_exact)
    error = torch.norm(u_exact - u_predicted)

    if norm_exact == 0:
        return torch.tensor(0.0, device=u_exact.device)
    return error / norm_exact

def relative_linf_error(u_exact, u_predicted):
    """Compute the relative L-infinity error."""
    norm_exact = torch.norm(u_exact, p=float('inf'))
    error = torch.norm(u_exact - u_predicted, p=float('inf'))

    if norm_exact == 0:
        return torch.tensor(0.0, device=u_exact.device)
    return error / norm_exact


def test(V_NN, num_points, dim, T, mu):
    """Evaluate V_NN against the exact solution at selected time points."""
    # Adapt num_points based on dimension to avoid OOM
    # For dim=50, use at most 5000 points
    actual_points = min(num_points, max(1000, 50000 // dim))

    x = 4 * torch.rand(actual_points, dim, device=device) - 2

    with torch.no_grad():
        V_NN.eval()
        for t_val in [0.0, 0.5, 1.0]:
            t = t_val * torch.ones(actual_points, 1, device=device)

            v_pred = V_NN(t, x)
            v_exact = V_exact(x, T, mu, dim, t=t_val)

            err_l2 = relative_error(v_exact, v_pred)
            err_linf = relative_linf_error(v_exact, v_pred)

            print(f"Time t = {t_val:.1f}")
            print(f"  Relative L2 Error    = {err_l2.item():.4e}")
            print(f"  Relative Linf Error  = {err_linf.item():.4e}\n")

    # Free memory after the test
    del x, t, v_pred, v_exact
    gc.collect()
    torch.cuda.empty_cache()

def W_gem(an, G_history, G_k, x0_ini, t_grid):
    """Compute Wasserstein distance between population and best response."""
    # Use fewer points for x0 in high dimension
    x0_subset = x0_ini[:min(1000, len(x0_ini))]

    W_k = wasserstein_fp(an, G_history, G_k,
                         x0_subset, t_grid,
                         p=1, blur=0.5, device=device)

    print(f"k {an.n:2d} │ "
          f"L1={np.linalg.norm(W_k,1):.3e} │ "
          f"L2={np.linalg.norm(W_k,2):.3e} │ "
          f"L∞={W_k.max():.3e}")

    del x0_subset
    gc.collect()
    torch.cuda.empty_cache()

    return W_k.max()


data = np.load('data50.npz')
# Keep x0_ini on CPU to save GPU memory
x0_ini_cpu = torch.tensor(data['x'], dtype=torch.float32)
# Compute mu and std on CPU
mu = 0*x0_ini_cpu.mean(axis=0) + 0.1
std = 0*torch.sqrt(x0_ini_cpu.var(axis=0)) + torch.sqrt(torch.tensor(0.105))

def get_x0_gpu(subset_size=None):
    """Transfer x0 to GPU. If subset_size is specified, take a random subset."""
    if subset_size is not None and subset_size < len(x0_ini_cpu):
        indices = torch.randperm(len(x0_ini_cpu))[:subset_size]
        return x0_ini_cpu[indices].to(device)
    return x0_ini_cpu.to(device)

if __name__ == "__main__":

    Round = 0

    V_NN = V_Net(dim=args.dim, ns=args.ns_v, act_func=args.act_func_v, hh=args.hh,
                            device=device, psi_func=psi_func, TT=args.t_final).to(device)

    V_NN2 = V_Net(dim=args.dim, ns=args.ns_v, act_func=args.act_func_v, hh=args.hh,
                            device=device, psi_func=psi_func, TT=args.t_final).to(device)

    G_NN = G_Net(dim=args.dim, ns=args.ns_g, act_func=args.act_func_g, hh=args.hh,
                            device=device, mu=mu, std=std, TT=args.t_final).to(device)

    t = torch.linspace(args.t0, args.t_final, args.num_samples_hjb, device=device).unsqueeze(1)
    t_g = torch.linspace(args.t0, args.t_final, args.num_samples_gen, device=device).unsqueeze(1)
    tt = torch.linspace(args.t0, args.t_final, 21, device=device).unsqueeze(1)

    # Store PATHS to G_NN models instead of keeping them in memory
    G_NN_paths = []

    nmax = 20
    tol = 5e-6
    delta = 1 + tol

    for n in range(nmax):
        if delta <= tol:
            break

        print(f"\n Computing The Best Response {n}: \n")

        start_time = time.time()

############# Find V1 ############

        # Load G_NN models from disk for this iteration
        G_NN_list = get_G_NN_list_from_paths(G_NN_paths, args.dim, args.ns_g,
                                                args.act_func_g, args.hh, device, mu, std, args.t_final)

        # Transfer x0 to GPU temporarily for this iteration
        x0_gpu = get_x0_gpu()
        an = Analytic(G_NN_list, Round, n, x0_gpu, device, VV=1)

#===============================Initialization=================================#
        if n == 0:

            print('Approximate solution of MFG \n')
            V_NN, G_NN = Solve_MFG2(an, V_NN, G_NN, args.num_epoch_mfg, t, args.lr, args.num_samples_mfg, device)
            # Save initial G_NN to disk and store the path
            path = save_G_NN_to_disk(G_NN, Round, -1)  # -1 for initialization
            G_NN_paths.append(path)
            # Reload G_NN_list with the new model
            G_NN_list = [load_G_NN_from_disk(path, args.dim, args.ns_g,
                                                args.act_func_g, args.hh, device, mu, std, args.t_final)]
            # Recreate Analytic with the new list
            an = Analytic(G_NN_list, Round, n, x0_gpu, device, VV=1)
            test(V_NN, args.num_points_test, args.dim, T=1, mu=0.1)

#===================Generate data BVP==================================#

        clear_gpu_memory()  # Clean before data generation

        # Use 20000 particles for good precision
        n_particles = 20000
        an.precompute_mean(n_points=100, n_particles_precise=n_particles)

        data = generate_data_threaded(an, V_NN, args.num_samples_bvp, device)

#==============================Approximate V===================================#

        print('Approximate solution of BVP \n')

        V_NN = Approximate_v(an, V_NN, data, args.num_epoch_v, t, args.lr_v, args.num_samples_hjb, Round, device, weight_decay=args.weight_decay)
        an.clear_mean()

        # Free BVP data after training
        del data
        clear_gpu_memory()


#===========================Simulate Points for M==============================#

        t_tr, X_tr, t_OUT, X_OUT = sim_points(an, V_NN, args.num_samples_gen, args.N, args.t0, args.t_final, device)

#=============================Train Generator==============================#

        G_NN, G_NN_new = Train_Gen(an, G_NN, V_NN, t_tr, X_tr, X_OUT, args.num_epoch_gen, t_g, args.lr_g, args.num_samples_gen, Round, device)

        # Free simulation data after training the generator
        del t_tr, X_tr, t_OUT, X_OUT
        clear_gpu_memory()


#==============================Error===================================#

        test(V_NN, args.num_points_test, args.dim, T=1, mu=0.1)

        delta = W_gem(an, G_NN_list, G_NN_new, x0_gpu, tt)

############ Compute J ###########
        # IMPORTANT: Compute J BEFORE clearing G_NN_list, passing G_NN_new directly
        J = Comp_J0(an, tt, x0_gpu, V_NN, G_NN=G_NN_new)

#==============================Update list===================================#

        # Save G_NN_new to disk
        path = save_G_NN_to_disk(G_NN_new, Round, n)
        G_NN_paths.append(path)

        # IMPORTANT: Do not keep G_NN models in memory — they will be reloaded from disk
        # Free all G_NN models from the list
        while len(G_NN_list) > 0:
            model = G_NN_list.pop(0)
            del model
        del an
        # Free x0_gpu from the first phase before VV=2
        del x0_gpu
        clear_gpu_memory()


############# Find V2 ############

        print('\nFind alpha1\n')

        # Reload G_NN_list for VV=2 — OPTIMIZATION: only load the last model since VV=2 uses only G_NN_list[-1]
        last_path = G_NN_paths[-1] if G_NN_paths else None
        if last_path:
            G_NN_list = [load_G_NN_from_disk(last_path, args.dim, args.ns_g,
                                                args.act_func_g, args.hh, device, mu, std, args.t_final)]
        else:
            G_NN_list = []
        # Transfer x0 to GPU again for VV=2
        x0_gpu = get_x0_gpu()
        an = Analytic(G_NN_list, Round, n, x0_gpu, device, VV=2)

#===========================Generate data BVP==================================#

        n_particles = 20000
        an.precompute_mean(n_points=100, n_particles_precise=n_particles)
        data = generate_data_threaded(an, V_NN, args.num_samples_bvp, device)


#==============================Approximate V===================================#

        print('Approximate solution of BVP \n')

        V_NN2 = Approximate_v2(an, V_NN2, data, args.num_epoch_v, t, args.lr_v2, args.num_samples_hjb, Round, device, weight_decay=args.weight_decay)
        an.clear_mean()

        # Free data after training V_NN2
        del data
        # Free G_NN_list from memory properly
        while len(G_NN_list) > 0:
            model = G_NN_list.pop(0)
            del model
        del G_NN_list
        del an
        clear_gpu_memory()

########## Compute V0 ############

        V0 = Comp_V(x0_gpu, V_NN2)

        # Free x0_gpu after Comp_V
        del x0_gpu

########## Test ################

        exploi = abs(V0 - J)

        print("The Exploitibility is :", exploi)

        # Copy weights instead of deepcopy to save memory
        G_NN.load_state_dict(G_NN_new.state_dict())
        del G_NN_new
        clear_gpu_memory()
