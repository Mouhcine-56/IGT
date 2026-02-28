import multiprocessing
import numpy as np
import argparse
import time
from model import *
from src.solve import *
import sys
from src.problem.prb import Analytic


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
parser.add_argument('--t0', type=int, default=0, help='temps initial')
parser.add_argument('--t_final', type=int, default=1, help='temps final')
parser.add_argument('--N', type=int, default=20, help='Number of time step')
parser.add_argument('--dim', type=int, default=10, help='dimension')
parser.add_argument('--num_samples_hjb', type=int, default=512, help='Number of Samples for training HJB')
parser.add_argument('--num_samples_mfg', type=int, default=512, help='Number of Samples for training MFG')
parser.add_argument('--num_samples_bvp', type=int, default=64, help='Number of Samples for  BVP')
parser.add_argument('--num_samples_gen', type=int, default=256, help='Number of Samples for training Generator')
parser.add_argument('--num_samples_conv', type=int, default=512, help='Number of Samples for compute convolution')
parser.add_argument('--num_points_test', type=int, default=10000)
parser.add_argument('--Max_Round', type=float, default=2, help='Number of Round')
parser.add_argument('--num_epoch_hjb', type=float, default=1000, help='Number of training iterations for approximate HJB')
parser.add_argument('--num_epoch_mfg', type=float, default=1000, help='Number of training iterations for approximate MFG')
parser.add_argument('--num_epoch_v', type=float, default=1000, help='Number of training iterations for approximate V')
parser.add_argument('--num_epoch_gen', type=float, default=3000, help='Number of training iterations for approximate GEN')
parser.add_argument('--freq', type=float, default=1000)
parser.add_argument('--ns_v', default=128, help='Network size of V_net')
parser.add_argument('--ns_g', default=128, help='Network sizeof G_net')
parser.add_argument('--lr', default=1e-3, help='learning rate of MFG')
parser.add_argument('--lr_v', default=1e-3, help='learning rate of V')
parser.add_argument('--lr_v2', default=1e-3, help='learning rate of V')
parser.add_argument('--lr_g', default=1e-4, help='learning rate of G')
parser.add_argument('--betas', default=(0.5, 0.9), help='Adam only')
parser.add_argument('--weight_decay',default=1e-3)
parser.add_argument('--act_func_v',default= lambda x: torch.tanh(x), help='Activation function for v')
parser.add_argument('--act_func_g',default= lambda x: torch.relu(x), help='Activation function for g')
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

    return torch.zeros(xx_inp.size(0),device=device)

def V_exact(x,T, mu, dim, t):
    
    Pi_t = (np.exp(2*T - t) - np.exp(t)) / (np.exp(2*T - t) + np.exp(t))
    s_t =  - Pi_t * mu
    c_t =  dim/2 * Pi_t * mu**2
    
    out = 0.5 * Pi_t * torch.sum(x**2, dim=1, keepdim=True) + s_t * torch.sum(x, dim=1, keepdim=True) + c_t
    
    return out

def relative_error(u_exact, u_predicted):
    """Computes relative L2 error."""
    norm_exact = torch.norm(u_exact)  # L2 norm
    error = torch.norm(u_exact - u_predicted)
    
    if norm_exact == 0:
        return torch.tensor(0.0, device=u_exact.device)
    return error / norm_exact

def relative_linf_error(u_exact, u_predicted):
    """Computes relative Linf error."""
    norm_exact = torch.norm(u_exact, p=float('inf'))  # Linf norm
    error = torch.norm(u_exact - u_predicted, p=float('inf'))

    if norm_exact == 0:
        return torch.tensor(0.0, device=u_exact.device)
    return error / norm_exact


def test(V_NN, num_points, dim, T, mu):
    #x = torch.linspace(-2, 2, num_points, device=device).unsqueeze(1)
    x = 4*torch.rand(num_points, dim, device=device)-2
    
    with torch.no_grad():
        V_NN.eval()
        for t_val in [0.0, 0.5, 1.0]:
            t = t_val * torch.ones(num_points, 1, device=device)

            v_pred = V_NN(t, x)
            v_exact = V_exact(x, T, mu, dim, t=t_val)

            err_l2 = relative_error(v_exact, v_pred)
            err_linf = relative_linf_error(v_exact, v_pred)

            print(f"Time t = {t_val:.1f}")
            print(f"  Relative L2 Error    = {err_l2.item():.4e}")
            print(f"  Relative Linf Error  = {err_linf.item():.4e}\n")
            
def W_gem(an, G_history,  G_k, x0_ini, m0, t_grid):

    W_k = wasserstein_fp(an, G_history, G_k,
                         x0_ini, m0, t_grid,
                         p=1, blur=0.5, device=device)

    print(f"k {an.n:2d} │ "
          f"L1={np.linalg.norm(W_k,1):.3e} │ "
          f"L2={np.linalg.norm(W_k,2):.3e} │ "
          f"L∞={W_k.max():.3e}")

    
# data = np.load('data1d.npz')
# M0 = np.load('m01d.npz')
# m0 =  torch.tensor(M0['x'], dtype=torch.float32, device=device)
# x0_ini = torch.tensor(data['x'], dtype=torch.float32, device=device)
# mu = x0_ini.mean(axis=0)
# std = torch.sqrt(x0_ini.var(axis=0))


data = np.load('data10d.npz')
X0 = torch.tensor(data['x'], dtype=torch.float32, device=device)
M0 = np.load('m010d.npz')
m0 =  torch.tensor(M0['x'], dtype=torch.float32, device=device)
x0_ini = X0
mu = 0*x0_ini.mean(axis=0) + 0.1
std = 0*torch.sqrt(x0_ini.var(axis=0)) + torch.sqrt(torch.tensor(0.105))



if __name__ == "__main__":
    
    V_NN = V_Net(dim=args.dim, ns=args.ns_v, act_func=args.act_func_v, hh=args.hh,
                            device=device, psi_func=psi_func, TT= args.t_final).to(device)
    
    V_NN2 = V_Net(dim=args.dim, ns=args.ns_v, act_func=args.act_func_v, hh=args.hh,
                            device=device, psi_func=psi_func, TT= args.t_final).to(device)
    
    G_NN = G_Net(dim=args.dim, ns=args.ns_g, act_func=args.act_func_g, hh=args.hh,
                            device=device, mu=mu, std=std, TT=args.t_final).to(device)

    t = torch.linspace(args.t0, args.t_final, args.num_samples_hjb, device=device).unsqueeze(1)   
    t_g = torch.linspace(args.t0, args.t_final, args.num_samples_gen, device=device).unsqueeze(1) 
    tt = torch.linspace(args.t0, args.t_final, 21, device=device).unsqueeze(1)

    
    for Round in range(0, args.Max_Round):
        
        print('Round: ', Round, '\n')
        
        G_NN_list = []
        
        if Round != 0:
            G_NN_list.append(copy.deepcopy(G_NN_new))
            
#         nmax=10
#         if Round == 2:
        nmax=10
        
        for n in range(nmax):
            
            print(f"\n Computing The Best Response {n}: \n")
            
            start_time = time.time()
        
############# Find V1 ############
        
            an = Analytic(G_NN_list, Round, n, x0_ini, device, VV=1)

            #test(V_NN, args.num_points_test, args.dim, T=1, mu=0.1)
#===============================Initialisation=================================#
            if Round ==0 and n == 0:
        
                #print('Approximate solution of MFG \n')
                #V_NN = Solve_HJB(an, V_NN, args.num_epoch_hjb, t, args.lr_v, args.num_samples_hjb, device)
                V_NN, G_NN = Solve_MFG2(an, V_NN, G_NN, args.num_epoch_mfg, t, args.lr, args.num_samples_mfg, device)
                G_NN_list.append(copy.deepcopy(G_NN))
                test(V_NN, args.num_points_test, args.dim, T=1, mu=0.1)
#===================Generate data BVP==================================#

            data = generate_data_threaded(an, V_NN, args.num_samples_bvp, device)

            
#==============================Approximate V===================================#

            print('Approximate solution of BVP \n')
            V_NN = Approximate_v(an, V_NN, data, args.num_epoch_v, t, args.lr_v, args.num_samples_hjb, Round, device)
        
        
        
#===========================Simulate Points for M==============================#

            t_tr, X_tr, t_OUT, X_OUT = sim_points(an, V_NN,  args.num_samples_gen, args.N, args.t0, args.t_final, device)
        
#=============================Train Geanerator==============================#
        
            G_NN, G_NN_new = Train_Gen(an, G_NN, V_NN, t_tr, X_tr, X_OUT, args.num_epoch_gen, t_g, args.lr_g, args.num_samples_gen, Round, device) 
        
        
#==============================Eroor===================================#
            
            test(V_NN, args.num_points_test, args.dim, T=1, mu=0.1)
            
            #wasserstein_population_vs_br(an, G_NN_list, G_NN_new, m0, tt, x0_ini, p=1)
            #wasserstein_population_vs_br_geom(an, G_NN_list, G_NN_new, m0, tt, x0_ini, p=1)
            
            W_gem(an, G_NN_list, G_NN_new, x0_ini, m0, tt)
#==============================Update list===================================#
        
            G_NN_list.append(copy.deepcopy(G_NN_new))
            torch.save(G_NN_new.state_dict(), f"G_NNN_round{Round}_n{n}.pth")
            
        
############ Compute J ###########
        
            J = Comp_J0(an, tt, x0_ini, V_NN)

        
############# Find V2 ############

            print('\nFind alpha1\n')


            an = Analytic(G_NN_list, Round, n, x0_ini, device, VV=2)
        
#===========================Generate data BVP==================================#
        
            data = generate_data_threaded(an, V_NN, args.num_samples_bvp, device)
            
        
#==============================Approximate V===================================#

            #print('Approximate solution of BVP \n')
            V_NN2 = Approximate_v2(an, V_NN2, data, args.num_epoch_v, t, args.lr_v2, args.num_samples_hjb, Round, device)
        
########## Compute V0 ############

            V0 = Comp_V(x0_ini, V_NN2)
        
########## Test ################

            exploi = abs(V0 - J)

            print("The Exploitibility is :", exploi)

            G_NN = copy.deepcopy(G_NN_new)
        
  

    
    
