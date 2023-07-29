# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Solve Cart Pole

# %%
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import tqdm

# %%
TENSORDTYPE = torch.float64

# %%
DEVICE = 'cuda:0'

# %%
LOGROOTDIR = './solve-cartpole-data/'

if not os.path.exists(LOGROOTDIR):
    os.mkdir(LOGROOTDIR)

# %%
# fixed seed for 3000 iteration
# seed = 8787     # for model pair (model_1, calc_loss_1)
# seed = 6150     # for moddel pair (model_1, calc_loss_1_off)
# seed = 4418     # for model pair (model_5, calc_loss_2)
# seed = 8173     # for model pair (model_5, calc_loss_2_off)
# seed = 1796     # for model pair (model_3, calc_loss_8)
# seed = 5325     # for model pair (model_3, calc_loss_8_off)

# random seed
# seed = random.randint(0, 10000)

# Set the random seed for Python's built-in random module
random.seed(seed)

# Set the random seed for NumPy
np.random.seed(seed)

# Set the random seed for PyTorch
torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

print(f"Seed: {seed}")


# %% [markdown]
# All problem-specific functions accpet tensor arguments.
#
# Take $f(t,x,a)$ as an example. We assume its inputs $t, x$ and $a$ are tensors with at least 1 dimension. All dimensions except the last are regarded as batch dimensions and are equivalent in function bodies. The outputs follow this rule too. This means even the time input $t$ and the value $f(t,x,a)$ are both scalar in their definition, we force them to be tensors in shape (1,) too.
#
# We also adopt the sequence-first convention, which is standard in seq2seq training. Most Tensors have shape (T, M, D), where
#
# - T : time axis
# - M : batch axis
# - D : original dim of this quantity

# %% [markdown]
# ## Helper Function

# %%
def re_cumsum(t, dim):
    r'''torch.cumsum in reverse direction'''
    return t + torch.sum(t, dim, keepdim=True) - torch.cumsum(t, dim)


# %% [markdown]
# Solve the stochastic optimal control problem Cart Pole (OpenAI Gym, see source code [here](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) and paper on the equation [here](https://coneural.org/florian/papers/05_cart_pole.pdf)):
#
# \begin{align}
# \operatorname{minimize}\quad& \operatorname{\mathbb{E}}\biggl[g(X_T)+\int_t^Tf(X_\tau,\alpha_\tau)\,d\tau\biggr], \\
# \operatorname{subject\ to}\quad& X_s = x + \int_t^s \bar{b}(\tau,X_\tau)\,d\tau +  \int_t^s\sigma(\hat{b}_0\,\alpha_\tau\,d\tau + dW_\tau),
# \end{align}
# where $x:=(x_c, \theta, \dot{x}_c, \dot\theta)$, $g(x)=x^\intercal Q x$, $f(x,a)=g(x) + \frac{r_0}{2}\|a\|^2$,
# $$\bar{b}(t,x) = 
#   \begin{bmatrix}
#   \dot{x_c} \\
#   \dot{\theta} \\
#   \frac{m_p l \dot{\theta}^2 \sin \theta}{m_c + m_p}
#   -\frac{m_p l \cos \theta}{m_c + m_p} \times
#     \frac{g \sin \theta - m_p l \dot{\theta}^2 \sin \theta \cos \theta
#       / (m_c + m_p)}{l\biggl(\frac{4}{3} - \frac{m_p \cos^2 \theta}{
#         m_c + m_p}\biggr)} \\
#   \frac{g \sin \theta - m_p l \dot{\theta}^2 \sin \theta \cos \theta
#     / (m_c + m_p)}{l\biggl(\frac{4}{3} - \frac{m_p \cos^2 \theta}{
#       m_c + m_p}\biggr)}
#   \end{bmatrix},
# \quad
# \sigma\hat{b}_0(t,x) = 
# \begin{bmatrix}
#   0 \\
#   0 \\
#   \frac{1}{m_c + m_p} + \frac{m_p l \cos \theta}{m_c + m_p} \times
#     \frac{\cos \theta / (m_c + m_p)}{l\biggl(\frac{4}{3}
#       -\frac{m_p \cos^2 \theta}{m_c + m_p}\biggr)} \\
#   \frac{- \cos \theta / (m_c + m_p)}{l\biggl(\frac{4}{3}
#       -\frac{m_p \cos^2 \theta}{m_c + m_p}\biggr)}
#   \end{bmatrix}.
# $$
# Choice I: $\sigma=\operatorname{diag}\{\sigma_1, \sigma_2, \sigma_3, \sigma_4\}$,
# $$\mu(t,x,z) = \operatorname*{arginf}_{a\in [a_\textrm{max},a_\textrm{min}]} \langle \hat{b}_0,z\rangle a + \frac{r_0}{2} a^2
# =\mathtt{clamp}(-\frac{\langle \hat{b}_0,z\rangle}{r_0},\mathtt{min=}a_{\mathrm{min}},\mathtt{max=}a_{\mathrm{max}}).$$

# %% [markdown]
# ## TODO
#
# Tips
#
# - [ ] Generate a dataset first and then try to optimize the loss over this dataset
# - [ ] float64 is way better than float32
# - [ ] Why the improved policy varies with the random seed and optimizer parameters? In fact, the solution $Y$ and $Z$ vary with the random seed and optimizer parameters. This may due to the nature of the BML loss, which could be non-convex and have very complex loss surface.
# - [ ] How the control bound affect the policy iteration algorithm? Clearly, it affects the policy improvement step. However, does it affect the policy evaluation of a given control policy? No. As long as the considered control policy remains withint the control bound, the policy evaluation result would be independent with the control bound. In this case, would it hold for all control bound that any improved policy based on it is a better policy? Yes, see the discussion at the begining of section IV.C (Noting that the old and the new policy are both constrained within the control bound.)
#
# Target
#
# - [ ] Inverted Pendulum
#   - [x] Problem statement
#   - [ ] On-policy iteration
#   - [ ] Off-policy iteration
# - [ ] Cart Pole
#   - [ ] Problem statement
#   - [ ] On-policy iteration
#   - [ ] Off-policy iteration

# %% [markdown]
# ## Problem

# %% [markdown]
# ## On-policy iteration

# %% [markdown]
# For any input policy $\alpha$, solve the decoupled linear FBSDE
# $$ \begin{aligned}
# X_s &= x + \int_t^s\bar{b}(\tau, X_\tau)\,d\tau + \int_t^s\sigma(\hat{b}_0\,\alpha_\tau\,d\tau + dW_\tau),\\
# Y_s &= g(X_T) + \int_s^Tf(X_\tau,\alpha_\tau)\,d\tau - \int_s^TZ_\tau^\intercal\,dW_\tau,
# \end{aligned}$$
# where $Z$ is approximated by a function $z(\cdot,\cdot)$.
#
# Output the improved policy $\alpha'(t, x) = \mu(t, x, z(t, x))$.

# %%
class ExampleCartPole(object):

    n = 4                       # state dim 
    udim = 1                    # control dim
    wdim = 4                    # noise dim

    def __init__(self):
        self.H = 75             # step nums
        self.dt = 0.02          # step size
        self.m = 16             # batch size

        self.x0 = torch.tensor([0., np.pi, 0., 0.], dtype=TENSORDTYPE, device=DEVICE)

        self.Umax = 20.
        self.r0 = .2
        self.Q_diag = torch.tensor([0., 10.01, 0.51, 0.51], dtype=TENSORDTYPE, device=DEVICE).reshape(1, 1, -1)
        self.sigma_diag = torch.tensor([0.01, 0.01, 0.01, 0.01], dtype=TENSORDTYPE, device=DEVICE).reshape(1, 1, -1)
        self.sigma_0 = self.sigma_diag
        
        self.ctrl = lambda t, x: torch.zeros_like(x[..., 0:1])

        # constants in the dynamic equation
        self.gravity = 9.8
        self.m_c = 1.0
        self.m_p = 0.1
        self.h_l = 0.5

        self.m_t = self.m_c + self.m_p  # total mass
        self.m_t_inv = 1 / self.m_t
        self.polemass_length = self.m_p * self.h_l

    def hat_b_0(self, t, x):
        return self.get_G(x) / self.sigma_diag

    def mu(self, t, x, z):
        hat_b_0 = self.hat_b_0(t, x)
        a = -torch.sum(z * hat_b_0, dim=-1, keepdim=True) / self.r0
        return torch.clamp(a, min=-self.Umax, max=self.Umax)

    def f(self, t, x, a):
        x = self.wrap_state(x)
        return .5 * self.r0*torch.sum(torch.square(a), -1, keepdim=True) + torch.sum(
            x*x*self.Q_diag, -1, keepdim=True)

    def g(self, x):
        x = self.wrap_state(x)
        return torch.sum(x*x*self.Q_diag, -1, keepdim=True)

    def step(self, t, x, a, dw):
        x_c, theta, xdot_c, thetadot = [x[..., i:i+1] for i in range(4)]

        sintheta, costheta = torch.sin(theta), torch.cos(theta)
        tmp = (
            a + self.polemass_length * thetadot**2 * sintheta
        ) / self.m_t
        thetaacc = (self.gravity * sintheta - costheta * tmp ) / (
            self.h_l * (4.0 / 3.0 - self.m_p * costheta**2 / self.m_t)
        )
        xacc = tmp - self.polemass_length * thetaacc * costheta / self.m_t
        
        return x + torch.cat([xdot_c, thetadot, xacc, thetaacc], dim=-1)*self.dt + self.sigma_diag * dw

    @staticmethod
    def wrap_state(x):
        x_c, theta, xdot_c, thetadot = [x[..., i] for i in range(4)]
        theta = (theta + np.pi) % (2*np.pi) - np.pi
        return torch.stack([x_c, theta, xdot_c, thetadot], dim=-1)

    def get_F(self, x):
        '''x should be 2d tensor with shape (batch_size, 4)'''
        x_c, theta, xdot_c, thetadot = [x[..., i] for i in range(4)]
        F_4 = (self.gravity * torch.sin(theta) - self.polemass_length *
               thetadot * thetadot * torch.sin(2 * theta) * .5 *
               self.m_t_inv) / (
                   self.h_l * (4 / 3 - self.m_p * self.m_t_inv *
                               .5 * (torch.cos(2 * theta) + 1)))
        F_3 = self.polemass_length * self.m_t_inv * (
            thetadot * thetadot * torch.sin(theta) - torch.cos(theta)
            * F_4)
        return torch.stack([xdot_c, thetadot, F_3, F_4], dim=-1)

    def get_G(self, x):
        '''x should be 2d tensor with shape (batch_size, 4)'''
        x_c, theta, xdot_c, thetadot = [x[..., i] for i in range(4)]
        G_4 = - torch.cos(theta) * self.m_t_inv / (
            self.h_l * (4 / 3 - self.m_p * self.m_t_inv *
                        .5 * (torch.cos(2 * theta) + 1)))
        G_3 = self.m_t_inv - G_4 * (
            self.m_t_inv * self.polemass_length * torch.cos(theta))
        return torch.stack([torch.zeros_like(G_3), torch.zeros_like(G_4),
                            G_3, G_4], dim=-1)

    @torch.no_grad()
    def sample_data(self, t0=0., x0=None, *, batch_num=None, ctrl=None):
        ctrl = ctrl or self.ctrl
        m = batch_num or self.m
        x0 = x0 or self.x0
        
        # discrete time points
        t = t0 + torch.tensor([self.dt*i for i in range(1+self.H)], dtype=TENSORDTYPE, device=DEVICE,
                             ).reshape(-1,1,1).expand(-1, m, 1)

        # sample noise dW
        dW = torch.normal(0., np.sqrt(self.dt), size=(self.H+1, m, self.wdim), dtype=TENSORDTYPE, device=DEVICE)
        # there are only H time interval and dW[-1] is only a placeholder.
        dW[-1] = 0.

        # simulate forward process with Euler scheme
        X = torch.empty(1+self.H, m, self.n, dtype=TENSORDTYPE, device=DEVICE)
        U = torch.empty(1+self.H, m, self.udim, dtype=TENSORDTYPE, device=DEVICE)
        X[0] = x0
        U[0] = ctrl(t[0], X[0])
        for i in range(self.H):
            X[i+1] = self.step(t[i], X[i], U[i], dW[i])
            U[i+1] = ctrl(t[i+1], X[i+1])

        f_s = self.f(t, X, U)
        f_s[-1] = self.g(X[-1])

        R = re_cumsum(f_s[:-1], dim=0) * self.dt + f_s[-1]
        R = torch.cat([R, f_s[-1:]], dim=0)
        
        return (
            t, X, U, f_s, R, dW,
        )
    
    @torch.no_grad()
    def off_process_data(self, current_ctrl, t_series, x_series, u_series, g_series, r_series, dw_series):
        U = current_ctrl(t_series, x_series)
        
        f_s = self.f(t_series, x_series, U)
        f_s[-1] = self.g(x_series[-1])

        R = re_cumsum(f_s[:-1], dim=0) * self.dt + f_s[-1]
        R = torch.cat([R, f_s[-1:]], dim=0)

        hat_b_0 = self.hat_b_0(t_series, x_series)
        b_series = hat_b_0*(U - u_series)
        
        return (
            t_series, x_series, b_series, f_s, R, dw_series
        )


# %%
# test sample data
fbsde = ExampleCartPole()
fbsde.m = 16

for _ in tqdm.trange(50):
    data = fbsde.sample_data()
    t, x, u, g, r, dW = data

    err = r - torch.stack([g[-1] + fbsde.dt * torch.sum(g[i:-1], dim=0) for i in range(g.shape[0])])
    err = err.abs().max()
    
    if err > 1e-12:
        break

    X = torch.empty_like(x)
    U = torch.empty_like(u)
    X[0] = x[0]
    U[0] = fbsde.ctrl(t[0], X[0])
    for i in range(fbsde.H):
        X[i+1] = X[i] + fbsde.dt * (fbsde.get_F(X[i]) + fbsde.get_G(X[i]) * U[i]) + dW[i] * fbsde.sigma_diag.squeeze(0)
        U[i+1] = fbsde.ctrl(t[i+1], X[i+1])

    err = (X - x).abs().max()

    if err > 1e-12:
        break

for s in data:
    print(s.shape)


# %% [markdown]
# # Basic Nets

# %%
class FCNet2(torch.nn.Module):

    def __init__(self, xdim, wdim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear1 = torch.nn.Linear(1+xdim, hidden_size, bias=True)
        self.head = torch.nn.Linear(hidden_size, wdim)
        self.activation = torch.nn.ReLU()
        self.to(dtype=TENSORDTYPE, device=DEVICE)
        
    def init_parameters_(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.normal_(self.linear1.bias)
        torch.nn.init.xavier_uniform_(self.head.weight)
        torch.nn.init.normal_(self.head.bias)
        
    def forward(self, t, x):
        z = torch.cat([1-t, x], dim=-1)
        z = self.activation(self.linear1(z))
        return self.head(z)
    
    @torch.no_grad()
    def get_derivative(self, t, x):
        z = self.linear1(torch.cat([1-t, x], dim=-1))
        grad = (self.linear1.weight.T * (z.unsqueeze(-2) > 0.) @ self.head.weight.T).squeeze(-1)
        return grad[...,1:]


# %%
class FCNet2_BOUND(torch.nn.Module):

    def __init__(self, xdim, wdim, hidden_size, bound):
        super().__init__()
        self.hidden_size = hidden_size
        self.bound = bound
        self.linear1 = torch.nn.Linear(1+xdim, hidden_size, bias=True)
        self.head = torch.nn.Linear(hidden_size, wdim)
        self.activation = torch.nn.ReLU()
        self.to(dtype=TENSORDTYPE, device=DEVICE)
        
    def init_parameters_(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.normal_(self.linear1.bias)
        torch.nn.init.xavier_uniform_(self.head.weight)
        torch.nn.init.normal_(self.head.bias)
        
    def forward(self, t, x):
        z = torch.cat([1-t, x], dim=-1)
        z = self.activation(self.linear1(z))

        # force boundness
        z = torch.clamp(z, min=-self.bound, max=self.bound)
        return torch.clamp(self.head(z), min=-self.bound, max=self.bound)


# %%
class FCNet3_BOUND(torch.nn.Module):

    def __init__(self, xdim, wdim, hidden_size, bound):
        super().__init__()
        self.hidden_size = hidden_size
        self.bound = bound
        self.linear1 = torch.nn.Linear(1+xdim, hidden_size, bias=True)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.head = torch.nn.Linear(hidden_size, wdim)
        self.activation = torch.nn.ReLU()
        self.to(dtype=TENSORDTYPE, device=DEVICE)
        
    def init_parameters_(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.normal_(self.linear1.bias)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.normal_(self.linear2.bias)
        torch.nn.init.xavier_uniform_(self.head.weight)
        torch.nn.init.normal_(self.head.bias)
        
    def forward(self, t, x):
        z = torch.cat([1-t, x], dim=-1)
        z = self.activation(self.linear1(z))

        # force boundness
        z = torch.clamp(z, min=-self.bound, max=self.bound)

        z = z + self.linear2(z)
        z = self.activation(z)

        # force boundness
        z = torch.clamp(z, min=-self.bound, max=self.bound)
        return torch.clamp(self.head(z), min=-self.bound, max=self.bound)


# %%
class TimeInvariant_Mixin(object):

    def forward(self, t, x):
        t = torch.zeros_like(t)
        return super().forward(t, x)


# %% [markdown]
# # Model

# %%
BUILDINGBLOCKS = {
    'FCNet2': FCNet2,
    'FCNet2_BOUND': FCNet2_BOUND,
    'FCNet3_BOUND': FCNet3_BOUND,

    # time invariant
    'FCNet2_TI': type('FCNet2_TI', (TimeInvariant_Mixin, FCNet2), {}),
    'FCNet2_Bound_TI': type('FCNet2_Bound_TI', (TimeInvariant_Mixin, FCNet2_BOUND), {}),
    'FCNet3_BOUND_TI': type('FCNet3_BOUND_TI', (TimeInvariant_Mixin, FCNet3_BOUND), {}),
}


# %%
class Model1(torch.nn.Module):
    
    def __init__(self, xdim, wdim, hidden_size, net_arch, **net_kws):
        super().__init__()
        self.znet = BUILDINGBLOCKS[net_arch](xdim, wdim, hidden_size, **net_kws)
        self.register_parameter("y0", torch.nn.Parameter(torch.tensor([1.])))
        
        self.to(dtype=TENSORDTYPE, device=DEVICE)
        
    def forward(self, t, x):
        return self.znet(t, x)
    
    
class Model2(torch.nn.Module):
    def __init__(self, xdim, wdim, hidden_size, net_arch, **net_kws):
        super().__init__()
        self.ynet = BUILDINGBLOCKS[net_arch](xdim, 1, hidden_size, **net_kws)
 
        self.to(dtype=TENSORDTYPE, device=DEVICE)
        
    def forward(self, t, x):
        return self.ynet(t, x)
    
    
class Model3(torch.nn.Module):
    def __init__(self, xdim, wdim, hidden_size, net_arch, **net_kws):
        super().__init__()
        self.ynet = BUILDINGBLOCKS[net_arch](xdim, 1, hidden_size, **net_kws)
        self.znet = BUILDINGBLOCKS[net_arch](xdim, wdim, hidden_size, **net_kws)
        
        self.to(dtype=TENSORDTYPE, device=DEVICE)
        
    def forward(self, t, x):
        return self.ynet(t, x), self.znet(t,x)
    

class Model5(torch.nn.Module):
    
    def __init__(self, xdim, wdim, hidden_size, net_arch, **net_kws):
        super().__init__()
        self.znet = BUILDINGBLOCKS[net_arch](xdim, wdim, hidden_size, **net_kws)

        self.to(dtype=TENSORDTYPE, device=DEVICE)

    def forward(self, t, x):
        return self.znet(t, x)


# %% [markdown]
# # Calculate Loss

# %% [markdown]
# Loss 1: Set (a)
# $$ \operatorname{\mathbb{E}} |\widetilde{Y}_t - y|^2. $$

# %%
def calc_loss_1(model, dt, t_series, x_series, u_series, g_series, r_series, dw_series):
    r'''Deep BSDE Loss'''
    z_series = model.znet(t_series, x_series)
    m0 = torch.sum(torch.sum(z_series*dw_series, dim=-1, keepdim=True), dim=0)
    loss = torch.square(r_series[0] - m0 - model.y0).mean()
    return loss

def calc_loss_1_off(model, dt, t_series, x_series, b_series, g_series, r_series, dw_series):
    z_series = model.znet(t_series, x_series)
    m0 = torch.sum(torch.sum(z_series*(dw_series-b_series*dt), dim=-1, keepdim=True)[:-1], dim=0)
    loss = torch.square(r_series[0] - m0 - model.y0).mean()
    return loss


# %% [markdown]
# Loss 2: Set (b)
# $$ \operatorname{\mathbb{E}} |\widetilde{Y}_t - \operatorname{\mathbb{E}}\widetilde{Y}_t|^2. $$

# %%
def calc_loss_2(model, dt, t_series, x_series, u_series, g_series, r_series, dw_series):
    r'''Deep BSDE-ML Loss'''
    z_series = model.znet(t_series, x_series)
    m0 = torch.sum(torch.sum(z_series*dw_series, dim=-1, keepdim=True), dim=0)
    loss = (r_series[0] - m0).var()
    return loss

def calc_loss_2_off(model, dt, t_series, x_series, b_series, g_series, r_series, dw_series):
    z_series = model.znet(t_series, x_series)
    m0 = torch.sum(torch.sum(z_series*(dw_series-b_series*dt), dim=-1, keepdim=True)[:-1], dim=0)
    loss = (r_series[0] - m0).var()
    return loss


# %% [markdown]
# Loss 7: Set (c)
# $$ \int_t^T\operatorname{\mathbb{E}} |R_s - \tilde{v}_s|^2\,ds. $$

# %%
def calc_loss_7(model, dt, t_series, x_series, u_series, g_series, r_series, dw_series):
    r'''Martingale Loss'''
    y_series = model.ynet(t_series, x_series)
    loss = torch.sum(torch.square(r_series - y_series).mean(dim=1, keepdim=True)) * dt
    return loss

def calc_loss_7_off(model, dt, t_series, x_series, b_series, g_series, r_series, dw_series):
    y_series = model.ynet(t_series, x_series)
    loss = torch.sum(torch.square(r_series - y_series).mean(dim=1, keepdim=True)) * dt
    return loss



# %% [markdown]
# Loss 8: Set (d)
# $$ \int_t^T\operatorname{\mathbb{E}} |\widetilde{Y}_s - \tilde{v}_s|^2\,ds. $$

# %%
def calc_loss_8(model, dt, t_series, x_series, u_series, g_series, r_series, dw_series):
    y_series, z_series = model(t_series, x_series)
    m_series = re_cumsum(torch.sum(z_series*dw_series, dim=-1, keepdim=True), 0)
    loss = torch.sum(torch.square(r_series - m_series - y_series).mean(dim=1, keepdim=True)) * dt
    return loss

def calc_loss_8_off(model, dt, t_series, x_series, b_series, g_series, r_series, dw_series):
    y_series, z_series = model(t_series, x_series)
    m_series = torch.sum(z_series*(dw_series-b_series*dt), dim=-1, keepdim=True)
    m_series = torch.cat([m_series[:-1], torch.zeros_like(m_series[-1:])], dim=0)
    m_series = re_cumsum(m_series, 0)
    loss = torch.sum((torch.square(r_series - m_series - y_series)).mean(dim=1, keepdim=True))*dt
    return loss


# %% [markdown]
# Other metric

# %%
def calc_metric_y(model, dt, t_series, x_series, u_series, g_series, r_series, dw_series):
    y0 = model.ynet(t_series[0], x_series[0]).mean()
    return (y0 - r_series[0].mean()).abs()

def calc_metric_z(model, dt, t_series, x_series, u_series, g_series, r_series, dw_series):
    z_series = model.znet(t_series, x_series)
    m0 = torch.sum(torch.sum(z_series*dw_series, dim=-1, keepdim=True), dim=0)
    loss = (r_series[0] - m0).var()
    return loss

def calc_impr_cost(model, sde):
    t, x, u, g, r, dW = sde.sample_data(batch_num=12800, ctrl=lambda t,x:sde.mu(t, x, model.znet(t, x)))
    return r[0].mean(), r[0].std()


# %% [markdown]
# # Policy Evaluation

# %%
def on_policy_evaluation(sde, calc_loss, model, optimizer, scheduler, *, max_epi=200, log_all=True):
    valid_data = sde.sample_data(batch_num=12800)
    
    para_log = []
    model.train(True)
    optimizer.zero_grad()
    
    for epi in range(max_epi):
        para_log.append({'grad step': epi})

        data = sde.sample_data()
        loss = calc_loss(model, sde.dt, *data)

        para_log[-1]['loss'] = loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.)

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad()

        if log_all:
            model.train(False)
            with torch.no_grad():
                if hasattr(model, 'ynet'):
                    para_log[-1]['metric_y'] = calc_metric_y(model, sde.dt, *valid_data).item()
                if hasattr(model, 'y0'):
                    para_log[-1]['metric_y'] = abs(model.y0.item() - valid_data[-2][0].mean().item() )
                if hasattr(model, 'znet'):
                    para_log[-1]['metric_z'] = calc_metric_z(model, sde.dt, *valid_data).item()

                    impr_cost_mean, impr_cost_std = calc_impr_cost(model, sde)
                    para_log[-1]['cost_mean'] = impr_cost_mean.item()
                    para_log[-1]['cost_std'] = impr_cost_std.item()
            model.train(True)

    return para_log


# %%
def off_policy_evaluation(sde, calc_loss, model, optimizer, scheduler, behavior_policy, *, max_epi=200, log_all=True):
    if hasattr(model, 'znet'):
        target_policy = lambda t, x:sde.mu(t, x, model.znet(t, x))
    else:
        target_policy = lambda t, x:sde.mu(t, x, model.ynet.get_derivative(t, x)*sde.sigma_0)
    
    para_log = []
    model.train(True)
    optimizer.zero_grad()
    
    for epi in range(max_epi):
        para_log.append({'grad step': epi})

        data = sde.sample_data(ctrl=behavior_policy)
        
        # process data with the target policy
        model.train(False)
        data = sde.off_process_data(target_policy, *data)
        model.train(True)
        
        loss = calc_loss(model, sde.dt, *data)

        para_log[-1]['loss'] = loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.)

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad()

        if log_all:
            model.train(False)
            with torch.no_grad():
                # valid data change with the target policy
                valid_data = sde.sample_data(batch_num=12800, ctrl=target_policy)
                if hasattr(model, 'ynet'):
                    para_log[-1]['metric_y'] = calc_metric_y(model, sde.dt, *valid_data).item()
                if hasattr(model, 'y0'):
                    para_log[-1]['metric_y'] = abs(model.y0.item() - valid_data[-2][0].mean().item() )
                if hasattr(model, 'znet'):
                    para_log[-1]['metric_z'] = calc_metric_z(model, sde.dt, *valid_data).item()

                    para_log[-1]['cost_mean'] = valid_data[-2][0].mean().item()
                    para_log[-1]['cost_std'] = valid_data[-2][0].std().item()
            model.train(True)

    return para_log


# %%
def create_optimizer(model, lr_configs):
    out = []
    if hasattr(model, 'y0'):
        out.append({'params': model.y0})
        out[-1].update(lr_configs['y0'])
    if hasattr(model, 'ynet'):
        out.append({'params': model.ynet.parameters(), })
        out[-1].update(lr_configs['znet'])
    if hasattr(model, 'znet'):
        out.append({'params': model.znet.parameters(), })
        out[-1].update(lr_configs['znet'])
    #return torch.optim.SGD(out)
    return torch.optim.Adam(out)


# %% [markdown]
# # Improve a given policy (on-policy & off-policy)
#
# Given a stochastic optimal control problem $\{\bar{b},\hat{b},\sigma,f,g,x_0,T\}$
# and an initial control policy $\alpha$,
# we adopt the following procedure to construct a better policy $\alpha'$.
#
# 1. *Collect data.* Generate a dataset by sending $\alpha$ into the system,
#    where each data point is a whole sample trajectory of the system.
# 2. *Optimize loss.* Train the model to optimize the given loss on the pregenerated
#    dataset.
# 3. *Construct return.* Construct the improved policy by calling the trained model.
#
# Let us elaborate on the *Optimize loss* step. It actually follows the standard pattern
# of supervised learning. The whole dataset contains many data points and are aranged as
# multiple mini batches. At each training step, only a batch of data points are sent to
# the model and contributed to the computation of loss function.

# %%
def on_policy_subroutine(sde, calc_loss, model, optimizer, scheduler, *,
                         batch_size=16, num_batches=1,
                         max_epoches=200, log_all=True, loss_scale=1.):
    valid_data = sde.sample_data(batch_num=12800)

    # collect data
    train_data = sde.sample_data(batch_num=batch_size * num_batches)
    def get_batch_data(i):
        batch_data = []
        for t in train_data:
            # slice the batch dimension
            batch_data.append(t[:, i*batch_size:(i+1)*batch_size, ...])
        return batch_data
    
    para_log = []
    model.train(True)
    optimizer.zero_grad()
    
    for epi in range(max_epoches):
        for step in range(num_batches):
            para_log.append({'grad step': epi*num_batches + step})

            para_log[-1]['epoch'] = epi
            para_log[-1]['step'] = step

            data = get_batch_data(step)
            loss = calc_loss(model, sde.dt, *data) * loss_scale

            para_log[-1]['loss'] = loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.)

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

        # average loss over the last epoch
        epo_losses = [p['loss'] for p in para_log[-num_batches:]]
        aver_epo_losses = sum(epo_losses) / num_batches
        for p in para_log[-num_batches:]:
            p['epoch training loss'] = aver_epo_losses

        if log_all:
            model.train(False)
            with torch.no_grad():
                if hasattr(model, 'ynet'):
                    para_log[-1]['metric_y'] = calc_metric_y(model, sde.dt, *valid_data).item()
                if hasattr(model, 'y0'):
                    para_log[-1]['metric_y'] = abs(model.y0.item() - valid_data[-2][0].mean().item() )
                if hasattr(model, 'znet'):
                    para_log[-1]['metric_z'] = calc_metric_z(model, sde.dt, *valid_data).item()

                    impr_cost_mean, impr_cost_std = calc_impr_cost(model, sde)
                    para_log[-1]['cost_mean'] = impr_cost_mean.item()
                    para_log[-1]['cost_std'] = impr_cost_std.item()
            model.train(True)

    return para_log


# %%
def off_policy_subroutine(sde, calc_loss, model, optimizer, scheduler, *, 
                          behavior_policy,
                          batch_size=16, num_batches=1, 
                          max_epoches=200, log_all=True, loss_scale=1.):

    # collect data
    train_data = sde.sample_data(ctrl=behavior_policy, batch_num=batch_size * num_batches)
    def get_batch_data(i):
        batch_data = []
        for t in train_data:
            # slice the batch dimension
            batch_data.append(t[:, i*batch_size:(i+1)*batch_size, ...])
        return batch_data

    para_log = []
    model.train(True)
    optimizer.zero_grad()
    
    for epi in range(max_epoches):
        for step in range(num_batches):
            para_log.append({'grad step': epi*num_batches + step})

            para_log[-1]['epoch'] = epi
            para_log[-1]['step'] = step

            data = get_batch_data(step)

            # process data with the target policy
            data = sde.off_process_data(sde.ctrl, *data)

            loss = calc_loss(model, sde.dt, *data) * loss_scale

            para_log[-1]['loss'] = loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.)

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

            # average loss over the last epoch
            epo_losses = [p['loss'] for p in para_log[-num_batches:]]
            aver_epo_losses = sum(epo_losses) / num_batches
            for p in para_log[-num_batches:]:
                p['epoch training loss'] = aver_epo_losses

        if log_all:
            model.train(False)
            with torch.no_grad():
                # valid data change with the target policy
                valid_data = sde.sample_data(batch_num=12800)
                if hasattr(model, 'ynet'):
                    para_log[-1]['metric_y'] = calc_metric_y(model, sde.dt, *valid_data).item()
                if hasattr(model, 'y0'):
                    para_log[-1]['metric_y'] = abs(model.y0.item() - valid_data[-2][0].mean().item() )
                if hasattr(model, 'znet'):
                    para_log[-1]['metric_z'] = calc_metric_z(model, sde.dt, *valid_data).item()

                    para_log[-1]['cost_mean'] = valid_data[-2][0].mean().item()
                    para_log[-1]['cost_std'] = valid_data[-2][0].std().item()
            model.train(True)

    return para_log


# %% [markdown]
# # Run

# %%
sde = ExampleCartPole()
sde.m = None
initial_ctrl = lambda t,x: -0.0*x[...,0:1]
initial_cost = sde.sample_data(ctrl=initial_ctrl, batch_num=12800)[-2][0].mean().item()
D = {
    
    'net_configs': {
        'xdim': sde.n,
        'wdim': sde.wdim,
        'hidden_size': 16,
        'net_arch': 'FCNet3_BOUND',
        'bound': 1000.,
    },
    'lr_configs': {
        'y0': {
            'lr': 100.5,
        },
        'ynet': {
            'lr': 0.01,
            'weight_decay': 1e-4,
#            'momentum': 1e-3,
#            'nesterov': True,
        },
        'znet': {
            'lr': 0.01,
            'weight_decay': 1e-4,
#            'momentum': 1e-3,
#            'nesterov': True,
        },
    },
    'scheduler_gamma': 0.999,
    'max_epoches': 50,
    'batch_size': 32,
    'num_baches': 4,
    'log_all': False,
    'loss_scale': 1.,
    'max_iteration': 3000,
    'inner_iter': 1,
}
model_loss_pairs = [
    [Model1, calc_loss_1],
#    [Model5, calc_loss_2],
#    [Model2, calc_loss_7],
#    [Model3, calc_loss_8],
#    [Model1, calc_loss_1_off],
#    [Model5, calc_loss_2_off],
#    [Model2, calc_loss_7_off],
#    [Model3, calc_loss_8_off],
]

cost_logs = [[] for _ in range(len(model_loss_pairs))]

for run_i in range(1):
    for pair_i, (model_cls, calc_loss) in enumerate(model_loss_pairs):
        # detect on-policy or off-policy
        mode = 'off-policy' if calc_loss.__name__.endswith('off') else 'on-policy'
      
        cost_logs[pair_i].append({'mode': mode, 'LossNum': calc_loss.__name__, 
                                  'run': run_i, 'iteration': 0, 'cost': initial_cost,})
      
        # set initial control
        sde.ctrl = initial_ctrl

        # track best model
        best_model = model_cls(**D['net_configs'])
        best_model.train(False)
        best_model_score = float('inf')
      
        paralogs = []
        for iter_i in tqdm.trange(1, D['max_iteration']+1):
            # to stable the output, we do several
            # policy evaluation in each iteration
            for inner_iter_i in range(D['inner_iter']):
                # policy evaluation: fit new model
                model = model_cls(**D['net_configs'])
                optimizer = create_optimizer(model, D['lr_configs'])
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=D['scheduler_gamma'])
                if mode == 'on-policy':
                    paralog = on_policy_subroutine(sde, calc_loss, model, optimizer, scheduler,
                                                   batch_size=D['batch_size'], num_batches=D['num_baches'],
                                                   max_epoches=D['max_epoches'],
                                                   log_all=D['log_all'], loss_scale=D['loss_scale'])
                elif mode == 'off-policy':
                    paralog = off_policy_subroutine(sde, calc_loss, model, optimizer, scheduler,
                                                   batch_size=D['batch_size'], num_batches=D['num_baches'],
                                                   behavior_policy=initial_ctrl,
                                                   max_epoches=D['max_epoches'],
                                                   log_all=D['log_all'], loss_scale=D['loss_scale'])

                # compare cost
                if hasattr(model, 'znet'):
                    current_cost = sde.sample_data(ctrl=lambda t, x:sde.mu(t, x, model.znet(t,x)), batch_num=12800)[-2][0].mean().item()
                else:
                    current_cost = sde.sample_data(ctrl=lambda t, x:sde.mu(t, x, model.ynet.get_derivative(t, x)*sde.sigma_0), batch_num=12800)[-2][0].mean().item()

                if current_cost < best_model_score:
                    best_model.load_state_dict(model.state_dict())
                    best_model_score = current_cost

                paralogs.append(paralog)
                cost_logs[pair_i].append({'mode': mode, 'LossNum': calc_loss.__name__, 
                                          'run': run_i, 'iteration': iter_i, 'inner iteration': inner_iter_i,
                                          'cost': current_cost,})

            # policy improvement: update control
            if hasattr(best_model, 'znet'):
                sde.ctrl = lambda t, x:sde.mu(t, x, best_model.znet(t, x))
            else:
                sde.ctrl = lambda t, x:sde.mu(t, x, best_model.ynet.get_derivative(t, x)*sde.sigma_0)
          
expI_cost_logs = cost_logs.copy()

# %%
ax = pd.DataFrame(paralogs[0]).plot(y=['loss', 'epoch training loss'])
ax.set_yscale('log')

# %%
fig, axes = plt.subplots(5, 2, figsize=(12, 12))
axes = axes.flatten()

# cost
real_cost = [expI_cost_logs[0][0]['cost']]
for p in expI_cost_logs[0][1:]:
    if p['cost'] > real_cost[-1]:
        real_cost.append(real_cost[-1])
    else:
        real_cost.append(p['cost'])
axes[0].plot([p['cost'] for p in expI_cost_logs[0]], marker='^')
axes[0].plot(real_cost, marker='s')
axes[0].set_title('COST')

if hasattr(best_model, 'znet'):
    t, x, u, g, r, dw = sde.sample_data(batch_num=128, ctrl=lambda t, x:sde.mu(t, x, best_model.znet(t, x)))
else:
    t, x, u, g, r, dw = sde.sample_data(batch_num=128, ctrl=lambda t, x:sde.mu(t, x, best_model.ynet.get_derivative(t, x)*sde.sigma_0))

# angle
axes[1].plot(x.detach().cpu()[:, :, 1].mean(dim=1), label='mean')
axes[1].plot(x.detach().cpu()[:, 0, 1], label='realization')
axes[1].set_title('ANGLE')
axes[1].legend()

# angular velocity
axes[2].plot(x.detach().cpu()[:, :, 3].mean(dim=-1))
axes[2].plot(x.detach().cpu()[:, 0, 3])
axes[2].set_title('OMEGA')

# control
axes[3].plot(u.detach().cpu()[:, :, 0].mean(dim=1))
axes[3].plot(u.detach().cpu()[:, 0, 0])
axes[3].set_title('CONTROL')

# loss
for it, paralog in enumerate(paralogs):
    axes[4].plot([p['loss'] for p in paralog], label=it)
axes[4].set_yscale('log')
# axes[4].legend()
axes[4].set_title('LOSS')

# Z
if hasattr(best_model, 'znet'):
    z = best_model.znet(t, x)
else:
    z = best_model.ynet.get_derivative(t, x)*sde.sigma_0

axes[5].plot(z.detach().cpu()[:, :, -1].mean(dim=1))
axes[5].plot(z.detach().cpu()[:, 0, -1])
axes[5].set_title('Z')

# cart position
axes[6].plot(x.detach().cpu()[:, :, 0].mean(dim=1), label='mean')
axes[6].plot(x.detach().cpu()[:, 0, 0], label='realization')
axes[6].set_title('POSITION')
axes[6].legend()

# cart velocity
axes[7].plot(x.detach().cpu()[:, :, 2].mean(dim=-1))
axes[7].plot(x.detach().cpu()[:, 0, 2])
axes[7].set_title('VELOCITY')

# real iteration cost
real_cost = [expI_cost_logs[0][0]['cost']]
real_iter = [0]
for i, p in enumerate(expI_cost_logs[0][1:]):
    if p['cost'] < real_cost[-1]:
        real_cost.append(p['cost'])
        real_iter.append(i+1)
axes[8].plot(real_iter, real_cost, marker='s')
axes[8].set_title('ITERCOST')

# real iteration loss
for it, paralog in enumerate(paralogs):
    if it + 1 in real_iter:
        axes[9].plot([p['loss'] for p in paralog], label=it)
axes[9].set_yscale('log')
axes[9].legend()
axes[9].set_title('ITERLOSS')

fig.tight_layout()

# %%
[expI_cost_logs[0][0]['cost']] + [min(p['cost'] for p in expI_cost_logs[0][1+i*D['inner_iter']:1+(i+1)*D['inner_iter']]) 
 for i in range(D['max_iteration'])]

# %%
r[0].mean()

# %% [markdown]
# # Save plots

# %%
# save cost dataframe
cost_df = pd.DataFrame(expI_cost_logs[0])
cost_df.to_csv(os.path.join(LOGROOTDIR, cost_df.iloc[0]['LossNum']+'.csv'), index=False)

# %%
# save trajectories
traj = {
    't': t, 'x': x, 'u': u,
    'g': g, 'r': r, 'dw': dw,
}

for key in traj:
    # move to cpu
    traj[key] = traj[key].detach().cpu()

torch.save(traj, os.path.join(LOGROOTDIR, cost_df.iloc[0]['LossNum']+'.pt'))

# save seed
with open(os.path.join(LOGROOTDIR, cost_df.iloc[0]['LossNum']+'.seed'), 'w') as fpr:
    fpr.write(str(seed))
