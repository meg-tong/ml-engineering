#%%
import numpy as np
import torch
from fancy_einsum import einsum
from torch import nn

import pytorch_utils

NUM_FREQUENCIES = 2
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6
#%%
x = np.linspace(-np.pi, np.pi, 2000)
y = TARGET_FUNC(x)

x_cos = np.array([np.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = np.array([np.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = np.random.randn()
A_n = np.random.randn(NUM_FREQUENCIES)
B_n = np.random.randn(NUM_FREQUENCIES)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):
    y_pred = 0.5 * a_0 + einsum('freq x, freq -> x', x_cos, A_n) + einsum('freq x, freq -> x', x_sin, B_n)
    loss = np.square(y - y_pred).sum()
    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])
        y_pred_list.append(y_pred)

    grad_y_pred = 2 * (y_pred - y)
    grad_a_0 = 0.5 * grad_y_pred.sum()
    grad_A_n = einsum('i j, j -> i', x_cos, grad_y_pred)
    grad_B_n = einsum('i j, j -> i', x_sin, grad_y_pred)

    a_0 -= LEARNING_RATE * grad_a_0
    A_n -= LEARNING_RATE * grad_A_n
    B_n -= LEARNING_RATE * grad_B_n
   
pytorch_utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)
# %%

x = torch.linspace(-torch.pi, torch.pi, 2000)
y = TARGET_FUNC(x)

x_cos = torch.stack([torch.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = torch.stack([torch.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = torch.randn((), dtype=torch.float)
A_n = torch.randn((NUM_FREQUENCIES), dtype=torch.float)
B_n = torch.randn((NUM_FREQUENCIES), dtype=torch.float)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):
    y_pred = 0.5 * a_0 + einsum('freq x, freq -> x', x_cos, A_n) + einsum('freq x, freq -> x', x_sin, B_n)
    loss = torch.square(y - y_pred).sum()
    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0.numpy(), A_n.numpy(), B_n.numpy()])
        y_pred_list.append(y_pred)

    grad_y_pred = 2 * (y_pred - y)
    grad_a_0 = 0.5 * grad_y_pred.sum()
    grad_A_n = einsum('i j, j -> i', x_cos, grad_y_pred)
    grad_B_n = einsum('i j, j -> i', x_sin, grad_y_pred)

    a_0 -= LEARNING_RATE * grad_a_0
    A_n -= LEARNING_RATE * grad_A_n
    B_n -= LEARNING_RATE * grad_B_n

pytorch_utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)
# %%
x = torch.linspace(-torch.pi, torch.pi, 2000)
y = TARGET_FUNC(x)

x_cos = torch.stack([torch.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = torch.stack([torch.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = torch.randn((), dtype=torch.float, requires_grad=True)
A_n = torch.randn((NUM_FREQUENCIES), dtype=torch.float, requires_grad=True)
B_n = torch.randn((NUM_FREQUENCIES), dtype=torch.float, requires_grad=True)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):
    y_pred = 0.5 * a_0 + einsum('freq x, freq -> x', x_cos, A_n) + einsum('freq x, freq -> x', x_sin, B_n)
    loss = torch.square(y - y_pred).sum()
    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0.detach().numpy(), A_n.detach().numpy(), B_n.detach().numpy()])
        y_pred_list.append(y_pred.detach().numpy())

    loss.backward()
    with torch.no_grad():
        a_0 -= LEARNING_RATE * a_0.grad
        A_n -= LEARNING_RATE * A_n.grad
        B_n -= LEARNING_RATE * B_n.grad
        a_0.grad = None
        A_n.grad = None
        B_n.grad = None

pytorch_utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)
# %% [markdown]
# Models
# %%
x = torch.linspace(-torch.pi, torch.pi, 2000)
y = TARGET_FUNC(x) 

x_cos = torch.stack([torch.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = torch.stack([torch.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

input = torch.transpose(torch.concat([x_cos, x_sin]), 0, 1)
model = nn.Linear(2 * NUM_FREQUENCIES, 1)

for step in range(TOTAL_STEPS):
    print(list(model.parameters())[0].detach().numpy())
    y_pred = model(input)
    loss = torch.square(y - y_pred).sum()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= LEARNING_RATE * param.grad
    model.zero_grad()

pytorch_utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)
# %%
x = torch.linspace(-torch.pi, torch.pi, 2000)
y = TARGET_FUNC(x) 

x_cos = torch.stack([torch.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = torch.stack([torch.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

input = torch.transpose(torch.concat([x_cos, x_sin]), 0, 1)
model = nn.Linear(2 * NUM_FREQUENCIES, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

for step in range(TOTAL_STEPS):
    print(list(model.parameters())[0].detach().numpy())
    y_pred = model(input)
    loss = torch.square(y - y_pred).sum()
    loss.backward()
    optimizer.step()
    model.zero_grad()

pytorch_utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)
# %%
