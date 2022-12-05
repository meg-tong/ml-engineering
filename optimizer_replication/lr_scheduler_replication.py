#%%
from typing import Callable

import torch as t

import optimizer_utils
import optimizer_replication
#%%
class ExponentialLR():
    def __init__(self, optimizer, gamma):
        '''Implements ExponentialLR.

        Like the PyTorch version, but assumes last_epoch=-1 and verbose=False
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html
        '''
        self.optimizer = optimizer
        self.gamma = gamma

    def step(self):
        self.optimizer.lr *= self.gamma

    def __repr__(self):
        return f"ExponentialLR(gamma={self.gamma})"

if __name__ == "__main__":
    optimizer_utils.test_ExponentialLR(ExponentialLR, optimizer_replication.SGD)
# %%
class StepLR():
    def __init__(self, optimizer, step_size, gamma=0.1):
        '''Implements StepLR.

        Like the PyTorch version, but assumes last_epoch=-1 and verbose=False
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
        '''
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.step_count = 1

    def step(self):
        if self.step_count % self.step_size == 0:
            self.optimizer.lr *= self.gamma
        self.step_count += 1

    def __repr__(self):
        return f"StepLR(step_size={self.step_size}),gamma={self.gamma})"

if __name__ == "__main__":
    optimizer_utils.test_StepLR(StepLR, optimizer_replication.SGD)
# %%
class MultiStepLR():
    def __init__(self, optimizer, milestones, gamma=0.1):
        '''Implements MultiStepLR.

        Like the PyTorch version, but assumes last_epoch=-1 and verbose=False
            https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html
        '''
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.step_count = 1

    def step(self):
        if self.step_count in self.milestones:
            self.optimizer.lr *= self.gamma
        self.step_count += 1

    def __repr__(self):
        return f"MultiStepLR(milestones={self.milestones}),gamma={self.gamma})"

if __name__ == "__main__":
    optimizer_utils.test_MultiStepLR(MultiStepLR, optimizer_replication.SGD)
# %%
def opt_fn_with_scheduler(
    fn: Callable, 
    xy: t.Tensor, 
    optimizer_class, 
    optimizer_kwargs, 
    scheduler_class = None, 
    scheduler_kwargs = dict(), 
    n_iters: int = 100
):
    '''Optimize the a given function starting from the specified point.

    scheduler_class: one of the schedulers you've defined, either ExponentialLR, StepLR or MultiStepLR
    scheduler_kwargs: keyword arguments passed to your optimiser (e.g. gamma)
    '''
    assert xy.requires_grad
    optimizer = optimizer_class([xy], **optimizer_kwargs)  
    scheduler = scheduler_class(optimizer, **scheduler_kwargs) if scheduler_class is not None else None
    xys = t.zeros((n_iters, 2))  
    for iter in range(n_iters):  
        xys[iter, :] = xy.detach()  
        z = fn(xy[0], xy[1])  
        z.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
    return xys.detach() 

if __name__ == "__main__":
    xy = t.tensor([-1.5, 2.5], requires_grad=True)
    x_range = [-2, 2]
    y_range = [-1, 3]
    optimizers = [
        (optimizer_replication.SGD, dict(lr=1e-3, momentum=0.98)),
        (optimizer_replication.SGD, dict(lr=1e-3, momentum=0.98)),
        (optimizer_replication.Adam, dict(lr=1e-3, betas=(0.8, 0.9))),
    ]
    schedulers = [
        (), # Empty list stands for no scheduler
        (ExponentialLR, dict(gamma=0.99)),
    ]
    fn = optimizer_replication.rosenbrocks_banana
    fig = optimizer_utils.plot_optimization_with_schedulers(opt_fn_with_scheduler, fn, xy, optimizers, schedulers, x_range, y_range, show_min=True)

    fig.show()
# %%
