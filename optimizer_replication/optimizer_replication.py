# %%
import optimizer_utils
import torch as t
from typing import Callable, Iterable, Tuple
# %%
def rosenbrocks_banana(x: t.Tensor, y: t.Tensor, a=1, b=100) -> t.Tensor:
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1

x_range = [-2, 2]
y_range = [-1, 3]
fig = optimizer_utils.plot_fn(rosenbrocks_banana, x_range, y_range, log_scale=True)
if __name__ == "__main__":
    fig.show()
# %%
def opt_fn_from_scratch(fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100):
    '''
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    '''
    assert xy.requires_grad
    xys = t.zeros((n_iters, 2))
    velocity = t.zeros_like(xy)
    for iter in range(n_iters):
        xys[iter, :] = xy.detach()
        z = fn(xy[0], xy[1])
        z.backward()
        with t.inference_mode():
            velocity *= momentum
            velocity += xy.grad
            xy -= lr * velocity
        xy.grad = t.zeros_like(xy)
    return xys.detach()

def opt_fn_with_sgd(fn: Callable, xy: t.Tensor, lr=0.001, momentum=0.98, n_iters: int = 100):
    assert xy.requires_grad  
    optimizer = t.optim.SGD([xy], lr=lr, momentum=momentum)  
    xys = t.zeros((n_iters, 2))  
    for iter in range(n_iters):  
        xys[iter, :] = xy.detach()  
        z = fn(xy[0], xy[1])  
        z.backward()
        optimizer.step()
        optimizer.zero_grad()
    return xys.detach()  

xy = t.tensor([-1.5, 2.5], requires_grad=True)
x_range = [-2, 2]
y_range = [-1, 3]

fig = optimizer_utils.plot_optimization_sgd(opt_fn_from_scratch, rosenbrocks_banana, xy, x_range, y_range, show_min=True)

if __name__ == "__main__":
    fig.show()
# %%
class SGD:
    params: list

    def __init__(self, params: Iterable[t.nn.parameter.Parameter], lr: float, momentum: float, weight_decay: float = 0.0):
        '''Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        '''
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [t.zeros_like(param) for param in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for param, velocity in zip(self.params, self.velocities):
            grad = param.grad + self.weight_decay * param
            velocity *= self.momentum
            velocity += grad
            param -= self.lr * velocity

    def __repr__(self) -> str:
        # Should return something reasonable here, e.g. "SGD(lr=lr, ...)"
        return f"SGD(lr={self.lr},momentum={self.momentum},weight_decay={self.weight_decay}"

if __name__ == "__main__":
    optimizer_utils.test_sgd(SGD)
#%%
class SGD_groups:

    def __init__(self, params, **kwargs):
        '''Implements SGD with momentum.

        Accepts parameters in groups, or an iterable.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        kwargs can contain lr, momentum or weight_decay
        '''
        if 'weight_decay' not in kwargs:
            kwargs['weight_decay'] = 0.0
        self.params = [list(group['params']) for group in params]
        all_params = [param  for params in self.params for param in params]
        if len(set(all_params)) != len(all_params):
            raise Exception()
        self.lrs = [group['lr'] if 'lr' in group else kwargs['lr'] for group in params] 
        self.momentums = [group['momentum'] if 'momentum' in group else kwargs['momentum']  for group in params] 
        self.weight_decays = [group['weight_decay'] if 'weight_decay' in group else kwargs['weight_decay'] for group in params]
        self.velocities = [[t.zeros_like(param) for param in params] for params in self.params]

    def zero_grad(self) -> None:
        for params in self.params:
            for param in params:
                param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        print(type(self.params))
        print(type(self.velocities))
        print(type(self.lrs))
        print(type(self.momentums))
        print(type(self.weight_decays))
        for params, velocities, lr, momentum, weight_decay in zip(self.params, self.velocities, self.lrs, self.momentums, self.weight_decays):
            for param, velocity in zip(params, velocities):
                grad = param.grad + weight_decay * param
                velocity *= momentum
                velocity += grad
                param -= lr * velocity

if __name__ == "__main__":
    optimizer_utils.test_sgd_param_groups(SGD_groups)
# %%
class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        alpha: float,
        eps: float,
        weight_decay: float,
        momentum: float,
    ):
        '''Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html#torch.optim.RMSprop
        '''
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.eps = eps
        self.velocities = [t.zeros_like(param) for param in self.params]
        self.velocities_sq = [t.zeros_like(param) for param in self.params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None
            
    @t.inference_mode()
    def step(self) -> None:
        for param, velocity, moving_average_sq in zip(self.params, self.velocities, self.velocities_sq):
            grad = param.grad + self.weight_decay * param
            moving_average_sq *= self.alpha
            moving_average_sq += (1 - self.alpha) * t.square(grad)
            velocity *= self.momentum
            velocity += grad / (t.sqrt(moving_average_sq) + self.eps)
            param -= self.lr * velocity

    def __repr__(self) -> str:
        return f"RMSProp(lr={self.lr},momentum={self.momentum},weight_decay={self.weight_decay},alpha={self.alpha},eps={self.eps}"

if __name__ == "__main__":
    optimizer_utils.test_rmsprop(RMSprop)
# %%
class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
        '''
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.eps = eps
        self.moving_averages = [t.zeros_like(param) for param in self.params]
        self.moving_averages_sq = [t.zeros_like(param) for param in self.params]
        self.t = 1

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for param, moving_average, moving_average_sq in zip(self.params, self.moving_averages, self.moving_averages_sq):
            grad = param.grad + self.weight_decay * param

            moving_average *= self.betas[0]
            moving_average += (1 - self.betas[0]) * grad
            moving_average_to_use = moving_average / (1 - self.betas[0] ** self.t)

            moving_average_sq *= self.betas[1]
            moving_average_sq += (1 - self.betas[1]) * t.square(grad)
            moving_average_sq_use = moving_average_sq / (1 - self.betas[1] ** self.t)

            param -= self.lr * moving_average_to_use / (t.sqrt(moving_average_sq_use) + self.eps)
        self.t += 1

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr},betas={self.betas},weight_decay={self.weight_decay},betas={self.betas},eps={self.eps}"

if __name__ == "__main__":
    optimizer_utils.test_adam(Adam)
# %%
def opt_fn(fn: Callable, xy: t.Tensor, optimizer_class, optimizer_kwargs, n_iters: int = 100):
    '''Optimize the a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
    optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    '''
    assert xy.requires_grad
    optimizer = optimizer_class([xy], **optimizer_kwargs)  
    xys = t.zeros((n_iters, 2))  
    for iter in range(n_iters):  
        xys[iter, :] = xy.detach()  
        z = fn(xy[0], xy[1])  
        z.backward()
        optimizer.step()
        optimizer.zero_grad()
    return xys.detach()  

xy = t.tensor([-1.5, 2.5], requires_grad=True)
x_range = [-2, 2]
y_range = [-1, 3]
optimizers = [
    (SGD, dict(lr=1e-3, momentum=0.98, weight_decay=0.0)),
    (SGD, dict(lr=5e-4, momentum=0.98, weight_decay=0.0)),
    (Adam, dict(lr=1e-3, betas=(0.8,0.98), eps=0.001, weight_decay=0.0)),
]
fn = rosenbrocks_banana

if __name__ == "__main__":
    fig = optimizer_utils.plot_optimization(opt_fn, fn, xy, optimizers, x_range, y_range)
    fig.show()
# %%
