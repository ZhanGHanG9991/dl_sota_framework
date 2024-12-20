import torch
import torch.nn as nn
from torchdiffeq import odeint

class VanDerPol(nn.Module):
    def __init__(self, alpha1, alpha2, W, alpha3, h_tmp):
        super(VanDerPol, self).__init__()
        self.h_tmp = None
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.W = W
        self.alpha3 = alpha3
        self.h_tmp = h_tmp
    
    def forward(self, t, hidden):
        if self.h_tmp is None:
            self.h_tmp = torch.zeros(hidden[0].size(0), hidden[0].size(1), hidden[0].size(2)).to(hidden[0].device)
        h1, h2 = hidden[0], hidden[1]
        dh1dt = self.alpha1 * h1 * (1 - h2**2) + self.alpha2 * h2 + self.W*h1 + self.alpha3 * self.h_tmp
        dh2dt = -h1
        return torch.stack([dh1dt, dh2dt], dim = 0)

class RNNwithODE(nn.Module):
    def __init__(self, configs):
        super(RNNwithODE, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_dim = configs.enc_in
        self.alpha1 = nn.Parameter(torch.tensor(configs.alpha1, dtype=torch.float32))
        self.alpha2 = nn.Parameter(torch.tensor(configs.alpha2, dtype=torch.float32))
        self.W = nn.Parameter(torch.tensor(configs.W, dtype=torch.float32))
        self.alpha3 = nn.Parameter(torch.tensor(configs.alpha3, dtype=torch.float32))
        self.h_tmp = None
        self.rnn = nn.RNN(input_size=self.in_dim, hidden_size=self.in_dim, batch_first=True)
        self.ode_func = VanDerPol(self.alpha1, self.alpha2, self.W, self.alpha3, self.h_tmp)   
        self.fc = nn.Linear(self.seq_len, self.pred_len)
    
    def input_to_hidden(self, x):
        h1_initial = torch.randn(x.size(0), self.seq_len, self.in_dim)
        h2_initial = torch.randn(x.size(0), self.seq_len, self.in_dim)
        hidden = torch.stack([h1_initial, h2_initial]).to(x.device)
        return hidden
    
    def forward(self, x, hidden = None):
        def print_grad_func(name):
            def hook(grad):
                print(name, grad)
            return hook
        if hidden is None:
            hidden = self.input_to_hidden(x)
        
        if self.h_tmp is None:
            self.h_tmp = torch.zeros(x.size(0), self.seq_len, self.in_dim).to(x.device)
        
        t_span = torch.linspace(0, 0.1, 5, device=hidden.device)
        self.h_tmp = x
        hidden[0], _ = self.rnn(x)
        hidden_out = odeint(self.ode_func, hidden, t_span, method='euler')
        hidden_concat = hidden_out[-1, 0].permute(2, 0, 1)
        # output [C, B, O]
        output = self.fc(hidden_concat)
        hidden = torch.clone(hidden_out[-1].detach())
        #ouput [B, O, C]
        # self.alpha1.register_hook(print_grad_func('alpha1'))
        output = output.permute(1, 2, 0)
        self.h_tmp = self.h_tmp.detach()

        return output, hidden

Model = RNNwithODE
