#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

SEED = 1

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_dtype(torch.float32)

data = pd.read_csv('porus_1K', sep='\\s+', header=None)
x_y = data.iloc[:, :2].values
X, Y = x_y[:, 0], x_y[:, 1]

K_uniform = 0.05
K = torch.tensor(K_uniform * np.ones(len(X)), dtype=torch.float32).cuda()

interior_pts = torch.tensor(x_y, dtype=torch.float32).to(DEVICE)
dirichlet_pts = torch.tensor(x_y[(Y == 0) | (X == 1) | (Y == 1)], dtype=torch.float32).to(DEVICE)
neumann_pts = torch.tensor(x_y[X == 0], dtype=torch.float32).to(DEVICE)

p_B = torch.tensor(1.0, dtype=torch.float32).to(DEVICE)
u_B = torch.tensor(0.0, dtype=torch.float32).to(DEVICE)
f = torch.tensor(1.0, dtype=torch.float32).to(DEVICE)


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 1024)
        self.fc7 = nn.Linear(1024, 512)
        self.fc8 = nn.Linear(512, 3)
        
        self.act = nn.GELU()
        
        self.skip1 = nn.Linear(2, 1024)
        self.skip2 = nn.Linear(1024, 1024)
        
        self.weight_params = nn.Parameter(torch.ones(4))
        self.register_buffer('base_scales', torch.tensor([100., 10., 50., 10.], dtype=torch.float32))
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h1 = self.act(self.fc1(x) + self.skip1(x))
        h2 = self.act(self.fc2(h1) + h1)
        h3 = self.act(self.fc3(h2) + self.skip2(h1))
        h4 = self.act(self.fc4(h3) + h2)
        h5 = self.act(self.fc5(h4) + h3)
        h6 = self.act(self.fc6(h5) + h4)
        h7 = self.act(self.fc7(h6))
        
        return self.fc8(h7)
    
    def get_weights(self):
        w = torch.sigmoid(self.weight_params) * self.base_scales
        return w[0], w[1], w[2], w[3]


def loss_function(model):
    x = interior_pts.clone().detach().requires_grad_(True)
    u_p = model(x)
    u, p = u_p[:, :2], u_p[:, 2]

    # PDE1: u + K ∇p = 0
    grad_p = autograd.grad(p.sum(), x, create_graph=True)[0]
    res    = u + K.view(-1,1) * grad_p
    pde1   = res.pow(2).mean()

    # PDE2: ∇·u = f
    du_x = autograd.grad(u[:, 0].sum(), x, create_graph=True)[0][:, 0]
    du_y = autograd.grad(u[:, 1].sum(), x, create_graph=True)[0][:, 1]
    div_u = du_x + du_y
    pde2 = (div_u - f).pow(2).mean()

    # Boundary conditions
    bcD = (model(dirichlet_pts)[:, 2] - p_B).pow(2).mean()
    uN = model(neumann_pts)[:, :2]
    bcN = (uN.matmul(torch.tensor([-1., 0.]).to(DEVICE)) - u_B).pow(2).mean()

    w1, w2, w3, w4 = model.get_weights()
    return w1 * pde1 + w2 * pde2 + w3 * bcD + w4 * bcN


def train_adam(model, optimizer, scheduler, epochs, print_every=500):
    for ep in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step(loss)
        
        if ep % print_every == 0:
            print(f"Epoch {ep:5d} | loss={loss.item():.3e}")


def train_forward_pinn():
    model = PINN().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=700, factor=0.5, verbose=False)

    start_adam = time.time()
    train_adam(model, optimizer, scheduler, epochs=15000, print_every=1000)

    start_lbfgs = time.time()
    get_lbfgs = lambda: optim.LBFGS(model.parameters(), lr=1.0, max_iter=100, 
                                    history_size=50, line_search_fn='strong_wolfe')

    for block in range(4000 // 100): 
        train_adam(model, optimizer, scheduler, epochs=100, print_every=100)

        lbfgs = get_lbfgs()
        
        def closure():
            lbfgs.zero_grad()
            loss = loss_function(model)
            loss.backward()
            return loss
        
        loss_lbfgs = lbfgs.step(closure)


    torch.save(model.state_dict(), "forward_output_1K_run1.pth")
    
    with torch.no_grad():
        preds = model(interior_pts).cpu().numpy()
    
    out = np.hstack((x_y, preds))
    np.savetxt("forward_output_1K_run1.csv", out, delimiter=",", header="x,y,u_x,u_y,p", comments="")

    
    return model

if __name__ == "__main__":
    train_forward_pinn()


# In[ ]:




