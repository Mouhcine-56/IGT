import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
bias_bool = True


# ===========================================
#              V_Net Architecture
# ===========================================
class V_Net(nn.Module):
    """
    Neural network approximator for the value function V(t, x).

    """

    def __init__(self, dim, ns, act_func, hh, device, psi_func, TT):
        super().__init__()

        self.lin1 = nn.Linear(dim + 1, ns, bias=bias_bool)
        self.lin2 = nn.Linear(ns, ns, bias=bias_bool)
        self.lin3 = nn.Linear(ns, ns, bias=bias_bool)
        self.linlast = nn.Linear(ns, 1)

        self.act_func = act_func
        self.psi_func = psi_func
        self.device = device

        self.dim = dim
        self.hh = hh
        self.TT = TT

    def forward(self, t, inp):
        """
        Forward evaluation with time interpolation between learned V(t, x) and terminal psi(x).
        """
        out = torch.cat((t, inp), dim=1)
        out = self.act_func(self.lin1(out))
        out = self.act_func(out + self.hh * self.lin2(out))
        out = self.act_func(out + self.hh * self.lin3(out))
        out = self.linlast(out)

        # Time interpolation between current and terminal state
        c1 = 1 - t.view(inp.size(0), 1)
        c2 = t.view(inp.size(0), 1)

        return c1 * out + c2 * self.psi_func(inp).view(inp.size(0), 1)

    def get_grad(self, t, inp):
        """
        Compute the gradient ∇_x V(t, x) using autograd.
        """
        self.eval()

        inp_tensor = torch.tensor(inp, dtype=torch.float32, device=self.device).clone().detach().requires_grad_(True)
        t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device).clone().detach()

        out = self(t_tensor, inp_tensor)

        grad_outputs = torch.ones_like(out)
        grad_inputs = torch.autograd.grad(out, inp_tensor, grad_outputs=grad_outputs, create_graph=True)

        return grad_inputs[0]  # ∇_x V(t, x)
        
# ===========================================
#              G_Net Architecture
# ===========================================
class G_Net(nn.Module):
    """
    Generator network that transports samples over time: G(t, x).
    Applies normalization to input and interpolates output over time.
    """

    def __init__(self, dim, ns, act_func, hh, device, mu, std, TT):
        super().__init__()

        self.lin1 = nn.Linear(dim + 1, ns)
        self.lin2 = nn.Linear(ns, ns)
        self.lin3 = nn.Linear(ns, ns)
        self.linlast = nn.Linear(ns, dim)

        self.act_func = act_func
        self.device = device
        self.mu = mu
        self.std = std

        self.dim = dim
        self.hh = hh
        self.TT = TT

    def forward(self, t, inp):
        """
        Forward pass through generator with input normalization and convex interpolation.
        """
        inp_normalized = (inp - self.mu.expand_as(inp)) * (1 / self.std.expand_as(inp))
        out = torch.cat((t, inp_normalized), dim=1)

        out = self.act_func(self.lin1(out))
        out = self.act_func(out + self.hh * self.lin2(out))
        out = self.act_func(out + self.hh * self.lin3(out))
        out = self.linlast(out)

        # Time-dependent interpolation between input and output
        ctt = t.view(inp.size(0), 1)
        c1 = ctt / self.TT
        c2 = (self.TT - ctt) / self.TT

        return c1 * out + c2 * inp

    def grad_t(self, t, x):
        """
        Compute the time derivative ∂G/∂t for each dimension using autograd.
        """
        G_out = self(t, x)
        G_t = torch.zeros_like(G_out)

        for i in range(G_out.shape[1]):
            grad_outputs = torch.zeros_like(G_out)
            grad_outputs[:, i] = 1.0

            grad = torch.autograd.grad(
                outputs=G_out,
                inputs=t,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            G_t[:, i] = grad.squeeze()

        return G_t

