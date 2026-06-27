#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
import os
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

data = pd.read_csv("forward_output_2K_run1.csv")
x_y = data[["x", "y"]].values
u_p_data = data[["u_x", "u_y", "p"]].values

interior_pts = torch.tensor(x_y, dtype=torch.float32).to(DEVICE)
u_p_forward = torch.tensor(u_p_data, dtype=torch.float32).to(DEVICE)

dirichlet_mask = (x_y[:, 1] == 0) | (x_y[:, 0] == 1) | (x_y[:, 1] == 1)
neumann_mask = (x_y[:, 0] == 0)

dirichlet_pts = torch.tensor(x_y[dirichlet_mask], dtype=torch.float32).to(DEVICE)
neumann_pts = torch.tensor(x_y[neumann_mask], dtype=torch.float32).to(DEVICE)

p_B = torch.tensor(1.0, dtype=torch.float32).to(DEVICE)
u_B = torch.tensor(0.0, dtype=torch.float32).to(DEVICE)
f = torch.tensor(1.0, dtype=torch.float32).to(DEVICE)

nvec = torch.tensor([-1.0, 0.0], dtype=torch.float32).to(DEVICE)


class InversePINN(nn.Module):
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
        
        self.K_net = nn.Sequential(
            nn.Linear(2, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        self.weight_params = nn.Parameter(torch.ones(5))
        self.register_buffer("base_scales", torch.tensor([300., 10., 10., 10., 1.], dtype=torch.float32))
        
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
    
    def get_K(self, x):
        return self.K_net(x)
    
    def get_weights(self):
        w = torch.sigmoid(self.weight_params) * self.base_scales
        return w[0], w[1], w[2], w[3], w[4]

    
current_epoch = 0

def loss_function(model: InversePINN):
    global current_epoch

    x = interior_pts.clone().detach().requires_grad_(True)
    u_p = model(x)
    u, p = u_p[:, :2], u_p[:, 2]

    # Data loss
    data_loss = torch.mean((u_p - u_p_forward) ** 2)

    # PDE1: u + K∇p = 0
    grad_p = autograd.grad(p.sum(), x, create_graph=True)[0]
    K_pred = model.get_K(x)
    pde1 = torch.mean((u + K_pred * grad_p) ** 2)

    # PDE2: ∇·u = f
    du_x = autograd.grad(u[:, 0].sum(), x, create_graph=True)[0][:, 0]
    du_y = autograd.grad(u[:, 1].sum(), x, create_graph=True)[0][:, 1]
    pde2 = torch.mean((du_x + du_y - f) ** 2)

    # Boundary conditions
    bcD = torch.mean((model(dirichlet_pts)[:, 2] - p_B) ** 2)
    uN = model(neumann_pts)[:, :2]
    bcN = torch.mean((uN.matmul(nvec) - u_B) ** 2)

    total_reg = torch.tensor(0.0, device=DEVICE)

    w1, w2, w3, w4, w5 = model.get_weights()
    total_loss = (w1 * data_loss + w2 * pde1 + w3 * pde2 + w4 * bcD + w5 * bcN + total_reg)

    return total_loss


def train_adam(model, optimizer, scheduler, epochs, print_every=500):
    global current_epoch
    for ep in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step(loss)
        current_epoch += 1

        if ep % print_every == 0:
            print(f"Epoch {current_epoch:5d} | loss={loss.item():.3e}")


def train_inverse_pinn():
    global current_epoch

    model = InversePINN().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=700, factor=0.5, verbose=False)

    current_epoch = 0

    t0 = time.time()
    train_adam(model, optimizer, scheduler, epochs=8000, print_every=1000)
    print(f"\nPhase A completed in {time.time() - t0:.1f}s")

    os.makedirs("ckpts_2knoreg", exist_ok=True)
    BEST_CKPT = "ckpts_2knoreg/BEST_MODEL.pth"
    best_loss = float("inf")

    get_lbfgs = lambda: optim.LBFGS(model.parameters(), lr=1.0, max_iter=100,
                                    history_size=50, line_search_fn="strong_wolfe")

    t1 = time.time()
    for block in range(4000 // 100): 
        train_adam(model, optimizer, scheduler, epochs=100, print_every=100)

        loss_adam = float(loss_function(model).detach())
        pre_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        lbfgs = get_lbfgs()

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            loss = loss_function(model)
            loss.backward()
            return loss

        lbfgs.step(closure)
        loss_post = float(loss_function(model).detach())

        if loss_post <= loss_adam:
            kept_loss = loss_post
            kept_tag = "kept LBFGS"
        else:
            model.load_state_dict(pre_state)
            kept_loss = loss_adam
            kept_tag = "reverted to AdamW"


        if kept_loss < best_loss:
            best_loss = kept_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, BEST_CKPT)

    print(f"\nPhase B completed in {time.time() - t1:.1f}s")

    if os.path.exists(BEST_CKPT):
        model.load_state_dict(torch.load(BEST_CKPT))

    torch.save(model.state_dict(), "inverse_output_2K_noreg_run1.pth")

    with torch.no_grad():
        preds_full = model(interior_pts).cpu().numpy()
        K_full = model.get_K(interior_pts).cpu().numpy().reshape(-1, 1)

    out = np.hstack((x_y, preds_full, K_full))
    np.savetxt("inverse_output_2K_noreg_run1.csv", out, delimiter=",", header="x,y,u_x,u_y,p,K", comments="")

    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(K_full.reshape(-1, 1))
    centers = sorted(kmeans.cluster_centers_.flatten())
    print(f"\n  Discovered low K:  {centers[0]:.6f}")
    print(f"  Discovered high K: {centers[1]:.6f}")
    print(f"  Separation: {centers[1] - centers[0]:.6f}")


    return model

if __name__ == "__main__":
    train_inverse_pinn()


# In[ ]:




