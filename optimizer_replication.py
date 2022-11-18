# %%
import arena_utils
import torch as t
from typing import Callable, Iterable, Tuple
# %%
def rosenbrocks_banana(x: t.Tensor, y: t.Tensor, a=1, b=100) -> t.Tensor:
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1

x_range = [-2, 2]
y_range = [-1, 3]
fig = arena_utils.plot_fn(rosenbrocks_banana, x_range, y_range, log_scale=True)
if __name__ == "__main__":
    fig.show()
# %%
def opt_fn_from_scratch(fn: Callable, xy: t.Tensor, lr=0.01, momentum=0.98, n_iters: int = 1000):
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
            velocity *= momentum + (1 - momentum) * xy.grad
            xy -= lr * velocity
        xy.grad = t.zeros_like(xy)
    return xys.detach()

def opt_fn_with_sgd(fn: Callable, xy: t.Tensor, lr=0.01, momentum=0.98, n_iters: int = 10):
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

fig = arena_utils.plot_optimization_sgd(opt_fn_with_sgd, rosenbrocks_banana, xy, x_range, y_range, n_iters=10, lr=0.01, momentum=0.98, show_min=True)

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
            param.grad = t.zeros_like(param)

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
    arena_utils.test_sgd(SGD)
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
            param.grad = t.zeros_like(param)

    @t.inference_mode()
    def step(self) -> None:
        for param, velocity, velocity_sq in zip(self.params, self.velocities, self.velocities_sq):
            grad = param.grad + self.weight_decay * param
            velocity_sq *= self.alpha
            velocity_sq += (1 - self.alpha) * t.square(grad)
            velocity *= self.momentum
            velocity += grad / (t.sqrt(velocity_sq) + self.eps)
            param -= self.lr * velocity

    def __repr__(self) -> str:
        return f"RMSProp(lr={self.lr},momentum={self.momentum},weight_decay={self.weight_decay},alpha={self.alpha},eps={self.eps}"

if __name__ == "__main__":
    arena_utils.test_rmsprop(RMSprop)
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
        self.velocities = [t.zeros_like(param) for param in self.params]
        self.velocities_sq = [t.zeros_like(param) for param in self.params]
        self.t = 1

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = t.zeros_like(param)

    @t.inference_mode()
    def step(self) -> None:
        for param, velocity, velocity_sq in zip(self.params, self.velocities, self.velocities_sq):
            grad = param.grad + self.weight_decay * param

            velocity *= self.betas[0]
            velocity += (1 - self.betas[0]) * grad
            velocity_to_use = velocity / (1 - self.betas[0] ** self.t)

            velocity_sq *= self.betas[1]
            velocity_sq += (1 - self.betas[1]) * t.square(grad)
            velocity_sq_use = velocity_sq / (1 - self.betas[1] ** self.t)

            param -= self.lr * velocity_to_use / (t.sqrt(velocity_sq_use) + self.eps)
        self.t += 1

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr},momentum={self.momentum},weight_decay={self.weight_decay},betas={self.betas},eps={self.eps}"

if __name__ == "__main__":
    arena_utils.test_adam(Adam)
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
]
fn = rosenbrocks_banana

if __name__ == "__main__":
    fig = arena_utils.plot_optimization(opt_fn, fn, xy, optimizers, x_range, y_range)
    fig.show()
# %%
