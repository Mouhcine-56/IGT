import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
bias_bool = True

class V_Net(torch.nn.Module):

    def __init__(self, dim, ns, act_func, hh, psi_func, device, TT):
        super().__init__()
        self.lin1 = torch.nn.Linear(dim+1, ns, bias=bias_bool)
        self.lin2 = torch.nn.Linear(ns, ns, bias=bias_bool)
        self.lin3 = torch.nn.Linear(ns, ns, bias=bias_bool)
        self.linlast = torch.nn.Linear(int(ns), 1)
        self.act_func = act_func
        self.psi_func = psi_func

        self.lintt = torch.nn.Linear(1, dim)

        self.dim = dim
        self.hh = hh
        self.TT = TT
        self.device = device

    def forward(self, t, inp):

        out = torch.cat((t, inp), dim=1)
        out = self.act_func(self.lin1(out))
        out = self.act_func(out + self.hh * self.lin2(out))
        out = self.act_func(out + self.hh * self.lin3(out))
        out = self.linlast(out)

        c1 = 1-torch.exp(t.view(inp.size(0), 1)-1)
        c2 = torch.exp(t.view(inp.size(0), 1)-1)
        
#         c1 = 1 - t.view(inp.size(0), 1)
#         c2 = t.view(inp.size(0), 1)
        return c1 * out + c2 * self.psi_func(inp).view(inp.size(0), 1)
    
    
    def get_grad(self, t, inp):
        
        self.eval()
        
        # Convert inputs to tensors and set requires_grad to True
        inp_tensor = torch.tensor(inp, dtype=torch.float32, device=self.device).clone().detach().requires_grad_(True)
        
        # Forward pass
        out = self(torch.tensor(t, dtype=torch.float32, device=self.device).clone().detach(), inp_tensor)
        
        # Backward pass to compute gradients
        grad_outputs = torch.ones_like(out)  # Assuming scalar output
        grad_inputs = torch.autograd.grad(out, inp_tensor, grad_outputs=grad_outputs, create_graph=True)
        
        return grad_inputs[0]  # Return gradient with respect to inp_tensor as numpy array