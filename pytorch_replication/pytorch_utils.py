import warnings
from typing import Callable, List, Optional

import numpy as np
import plotly.graph_objects as go
import plotly.graph_objs as go
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from einops import repeat
from plotly.subplots import make_subplots
from torch import nn
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import ipywidgets as wg

def test_relu(ReLU):
    x = t.randn(10) - 0.5
    actual = ReLU()(x)
    expected = F.relu(x)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_relu` passed!")

def test_flatten(Flatten):
    x = t.arange(24).reshape((2, 3, 4))
    assert Flatten(start_dim=0)(x).shape == (24,)
    assert Flatten(start_dim=1)(x).shape == (2, 12)
    assert Flatten(start_dim=0, end_dim=1)(x).shape == (6, 4)
    assert Flatten(start_dim=0, end_dim=-2)(x).shape == (6, 4)
    print("All tests in `test_flatten` passed!")

def test_linear_forward(Linear):
    """Your Linear should produce identical results to torch.nn given identical parameters."""
    x = t.rand((10, 512))
    yours = Linear(512, 64)
    assert yours.weight.shape == (64, 512), f"Linear layer weights have wrong shape: {yours.weight.shape}, expected shape = (64, 512)"
    assert yours.bias.shape == (64,), f"Linear layer bias has wrong shape: {yours.bias.shape}, expected shape = (64,)"
    official = t.nn.Linear(512, 64)
    yours.weight = official.weight
    yours.bias = official.bias
    actual = yours(x)
    expected = official(x)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_linear_forward` passed!")

def test_linear_parameters(Linear):
    m = Linear(2, 3)
    params = dict(m.named_parameters())
    assert len(params) == 2, f"Your model has {len(params)} recognized Parameters"
    assert list(params.keys()) == [
        "weight",
        "bias",
    ], f"For compatibility with PyTorch, your fields should be named weight and bias, not {tuple(params.keys())}"
    print("All tests in `test_linear_parameters` passed!")

def test_linear_no_bias(Linear):
    
    x = t.rand((10, 512))
    yours = Linear(512, 64, bias=False)

    assert yours.bias is None, "Bias should be None when not enabled."
    assert len(list(yours.parameters())) == 1

    official = nn.Linear(512, 64, bias=False)
    yours.weight = official.weight
    actual = yours(x)
    expected = official(x)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_linear_no_bias` passed!")


def visualize(dataloader):
    (sample, sample_labels) = next(iter(dataloader))
        
    fig = make_subplots(
        rows=2, cols=5,
        horizontal_spacing=0.02, vertical_spacing=0.02,
        subplot_titles=[str(label.item()) for label in sample_labels[:10]]
    )
    for row in range(2):
        for col in range(5):
            z = repeat((255 * (0.28 + 0.35*sample[5*row+col, 0])).numpy().astype(int), "h w -> h w 3")
            fig.add_trace(go.Image(z=z), row=row+1, col=col+1)
            
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_layout(margin=dict(t=50, b=0, r=20, l=20))

    fig.show(config={'displayModeBar': False})

def get_mnist(subsample: Optional[int] = None):
    """Return MNIST data using the provided Tensor class."""
    mnist_train = datasets.MNIST("../data", train=True, download=True)
    mnist_test = datasets.MNIST("../data", train=False)
    if subsample is None:
        subsample = 1
    print("Preprocessing data...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.28,), (0.35,))])
    train_indexes = range(0, len(mnist_train), subsample)
    train_reduced = [mnist_train[i] for i in train_indexes]
    train_tensors = torch.utils.data.TensorDataset(
        torch.stack([transform(img) for img, label in tqdm(train_reduced, desc="Training data")]),
        torch.tensor([label for img, label in train_reduced]),
    )

    test_indexes = range(0, len(mnist_test), subsample)
    test_reduced = [mnist_test[i] for i in test_indexes]
    test_tensors = torch.utils.data.TensorDataset(
        torch.stack([transform(img) for img, label in tqdm(test_reduced, desc="Test data")]),
        torch.tensor([label for img, label in test_reduced]),
    )

    train_loader = torch.utils.data.DataLoader(train_tensors, shuffle=True, batch_size=512)
    test_loader = torch.utils.data.DataLoader(test_tensors, batch_size=512)
    return train_loader, test_loader


def test_log_back(log_back):
    a = np.array([1, np.e, np.e**np.e])
    b = np.log(a)
    grad_out = np.array([2.0, 2.0, 2.0])
    actual = log_back(grad_out, b, a)
    expected = [2.0, 2.0 / np.e, 2.0 / (np.e**np.e)]
    np.testing.assert_allclose(actual, expected)
    print("All tests in `test_einsum_inner` passed!")

def test_unbroadcast(unbroadcast):
    small = np.ones((2, 1, 3))
    large = np.broadcast_to(small, (5, 1, 2, 4, 3))
    out = unbroadcast(large, small)
    assert out.shape == small.shape
    assert (out == 20.0).all(), "Each element in the small array appeared 20 times in the large array."

    small = np.ones((2, 1, 3))
    large = np.broadcast_to(small, (5, 1, 2, 1, 3))
    out = unbroadcast(large, small)
    assert out.shape == small.shape
    assert (out == 5.0).all(), "Each element in the small array appeared 5 times in the large array."

    small = np.ones((2, 1, 3))
    large = np.broadcast_to(small, (2, 4, 3))
    out = unbroadcast(large, small)
    assert out.shape == small.shape
    assert (out == 4.0).all(), "Each element in the small array appeared 4 times in the large array."
    print("All tests in `test_unbroadcast` passed!")

def test_multiply_back(multiply_back0, multiply_back1):
    a = np.array([1, 2, 3])
    b = np.array([2])
    c = a * b
    grad_out = np.array([2.0, 2.0, 2.0])
    actual = multiply_back0(grad_out, c, a, b)
    expected = np.array([4.0, 4.0, 4.0])
    assert actual.shape == expected.shape, f"Output has wrong shape: {actual.shape}, expected {expected.shape}"
    assert np.allclose(actual, expected), "Correct shape, but wrong values"
    actual = multiply_back1(grad_out, c, a, b)
    expected = np.array([12.0])
    assert actual.shape == expected.shape, f"Output has wrong shape: {actual.shape}, expected {expected.shape}"
    assert np.allclose(actual, expected), "Correct shape, but wrong values"
    print("All tests in `test_multiply_back` passed!")

def test_multiply_back_float(multiply_back0, multiply_back1):
    a = np.array([1, 2, 3])
    b = 2
    c = a * b
    grad_out = np.array([2.0, 2.0, 2.0])
    actual = multiply_back0(grad_out, c, a, b)
    expected = [4.0, 4.0, 4.0]
    np.testing.assert_allclose(actual, expected)
    a = np.array([1, 2, 3])
    b = 2
    c = a * b
    grad_out = np.array([2.0, 2.0, 2.0])
    actual = multiply_back1(grad_out, c, b, a)
    expected = [4.0, 4.0, 4.0]
    np.testing.assert_allclose(actual, expected)
    print("All tests in `test_multiply_back_float` passed!")

def test_forward_and_back(forward_and_back):
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 1])
    c = np.array([10])
    dg_da, dg_db, dg_dc = forward_and_back(a, b, c)
    expected_dg_da = np.array([1, 1 / 2, 1 / 3])
    expected_dg_db = np.array([1 / 2, 1 / 3, 1])
    expected_dg_dc = np.array([0.13028834])
    np.testing.assert_allclose(dg_da, expected_dg_da)
    np.testing.assert_allclose(dg_db, expected_dg_db)
    np.testing.assert_allclose(dg_dc, expected_dg_dc)
    print("All tests in `test_forward_and_back` passed!")

def test_back_func_lookup(BackwardFuncLookup):
    backward_funcs = BackwardFuncLookup()
    backward_funcs.add_back_func(np.log, 0, np.exp)
    assert backward_funcs.get_back_func(np.log, 0) == np.exp
    backward_funcs.add_back_func(np.multiply, 0, np.divide)
    assert backward_funcs.get_back_func(np.multiply, 0) == np.divide
    backward_funcs.add_back_func(np.multiply, 1, np.add)
    assert backward_funcs.get_back_func(np.multiply, 1) == np.add
    print("All tests in `test_back_func_lookup` passed!")

def test_log(Tensor, log_forward):
    a = Tensor([np.e, np.e**np.e], requires_grad=True)
    b = log_forward(a)
    np.testing.assert_allclose(b.array, [1, np.e])
    assert b.requires_grad == True, "Should require grad because input required grad."
    assert b.is_leaf == False
    assert b.recipe is not None
    assert len(b.recipe.parents) == 1 and b.recipe.parents[0] is a
    assert len(b.recipe.args) == 1 and b.recipe.args[0] is a.array
    assert b.recipe.kwargs == {}
    assert b.recipe.func is np.log
    c = log_forward(b)
    np.testing.assert_almost_equal(c.array, [0, 1])
    print("All tests in `test_log` passed!")

def test_log_no_grad(Tensor, log_forward):
    d = Tensor([1, np.e])
    e = log_forward(d)
    assert e.requires_grad == False, "Should not require grad because input did not."
    assert e.recipe is None
    np.testing.assert_allclose(e.array, [0, 1])
    print("All tests in `test_log_no_grad` passed!")

def test_multiply(Tensor, multiply):
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = Tensor([[0], [1], [10]], requires_grad=True)
    c = multiply(a, b)
    assert c.requires_grad == True, "Should require grad because input required grad."
    assert c.is_leaf == False
    assert c.recipe is not None
    assert len(c.recipe.parents) == 2 and c.recipe.parents[0] is a and c.recipe.parents[1] is b
    assert len(c.recipe.args) == 2 and c.recipe.args[0] is a.array and c.recipe.args[1] is b.array
    assert c.recipe.kwargs == {}
    assert c.recipe.func is np.multiply
    expected = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 10, 20, 30]])
    np.testing.assert_allclose(c.array, expected)
    print("All tests in `test_multiply` passed!")

def test_multiply_no_grad(Tensor, multiply):
    a = Tensor([0, 1, 2, 3], requires_grad=False)
    b = Tensor([[0], [1], [10]], requires_grad=False)
    c = multiply(a, b)
    assert c.requires_grad == False, "Should not require grad because input did not require grad."
    assert c.recipe is None
    expected = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 10, 20, 30]])
    np.testing.assert_allclose(c.array, expected)
    print("All tests in `test_multiply_no_grad` passed!")

def test_multiply_float(Tensor, multiply):
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = 3
    c = multiply(a, b)
    assert c.requires_grad == True
    assert c.recipe is not None
    assert len(c.recipe.parents) == 1 and c.recipe.parents[0] is a
    assert len(c.recipe.args) == 2 and c.recipe.args[0] is a.array and c.recipe.args[1] is b
    assert c.recipe.kwargs == {}
    assert c.recipe.func is np.multiply
    expected = np.array([0, 3, 6, 9])
    np.testing.assert_allclose(c.array, expected)
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = 3
    c = multiply(b, a)
    assert c.requires_grad == True
    assert c.recipe is not None
    assert len(c.recipe.parents) == 1 and c.recipe.parents[1] is a
    assert len(c.recipe.args) == 2 and c.recipe.args[0] is b and c.recipe.args[1] is a.array
    assert c.recipe.kwargs == {}
    assert c.recipe.func is np.multiply
    expected = np.array([0, 3, 6, 9])
    np.testing.assert_allclose(c.array, expected)
    print("All tests in `test_multiply_float` passed!")

def test_sum(wrap_forward_fn, Tensor):
    # This tests keyword arguments
    def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
        """Like torch.sum, calling np.sum internally."""
        return np.sum(x, axis=dim, keepdims=keepdim)
    global sum
    sum = wrap_forward_fn(_sum)
    a = Tensor(np.array([[0.0, 1.0], [2.0, 3.0]]), requires_grad=True)
    assert a.sum(0).shape == (2,)
    assert a.sum(0, True).shape == (1, 2)
    print("All tests in `test_sum` passed!")
    


class Node:
    def __init__(self, *children):
        self.children = list(children)


def get_children(node: Node) -> List[Node]:
    return node.children

def test_topological_sort_linked_list(topological_sort):
    z = Node()
    y = Node(z)
    x = Node(y)
    expected = [z, y, x]
    for e, a in zip(expected, topological_sort(x, get_children)):
        assert e is a
    print("All tests in `test_topological_sort_linked_list` passed!")

def test_topological_sort_branching(topological_sort):
    z = Node()
    y = Node()
    x = Node(y, z)
    w = Node(x)
    name_lookup = {w: "w", x: "x", y: "y", z: "z"}
    out = "".join([name_lookup[n] for n in topological_sort(w, get_children)])
    assert out == "zyxw" or out == "yzxw"
    print("All tests in `test_topological_sort_branching` passed!")

def test_topological_sort_rejoining(topological_sort):
    z = Node()
    y = Node(z)
    x = Node(y)
    w = Node(z, x)
    name_lookup = {w: "w", x: "x", y: "y", z: "z"}
    out = "".join([name_lookup[n] for n in topological_sort(w, get_children)])
    assert out == "zyxw"
    print("All tests in `test_topological_sort_rejoining` passed!")

def test_topological_sort_cyclic(topological_sort):
    z = Node()
    y = Node(z)
    x = Node(y)
    z.children = [x]
    try:
        topological_sort(x, get_children)
    except:
        assert True
    else:
        assert False
    print("All tests in `test_topological_sort_cyclic` passed!")

def test_backprop(Tensor):
    a = Tensor([np.e, np.e**np.e], requires_grad=True)
    b = a.log()
    c = b.log()
    c.backward(end_grad=np.array([1.0, 1.0]))
    assert c.grad is None
    assert b.grad is None
    assert a.grad is not None
    assert np.allclose(a.grad.array, 1 / b.array / a.array)
    print("All tests in `test_backprop` passed!")

def test_backprop_branching(Tensor):
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([1, 2, 3], requires_grad=True)
    c = a * b
    c.backward(end_grad=np.array([1.0, 1.0, 1.0]))
    assert np.allclose(a.grad.array, b.array)
    assert np.allclose(b.grad.array, a.array)
    print("All tests in `test_backprop_branching` passed!")

def test_backprop_requires_grad_false(Tensor):
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([1, 2, 3], requires_grad=False)
    c = a * b
    c.backward(end_grad=np.array([1.0, 1.0, 1.0]))
    assert np.allclose(a.grad.array, b.array)
    assert b.grad is None
    print("All tests in `test_backprop_requires_grad_false` passed!")

def test_backprop_float_arg(Tensor):
    a = Tensor([1, 2, 3], requires_grad=True)
    b = 2
    c = a * b
    d = 2
    e = d * c
    e.backward(end_grad=np.array([1.0, 1.0, 1.0]))
    assert e.grad is None
    assert c.grad is None
    assert a.grad is not None
    assert np.allclose(a.grad.array, np.array([4.0, 4.0, 4.0]))
    print("All tests in `test_backprop_float_arg` passed!")

def test_negative_back(Tensor):
    a = Tensor([-1, 0, 1], requires_grad=True)
    b = -a
    c = -b
    c.backward(end_grad=np.array([[1.0, 1.0, 1.0]]))
    assert a.grad is not None
    np.testing.assert_allclose(a.grad.array, [[1.0, 1.0, 1.0]])
    print("All tests in `test_negative_back` passed!")

def test_exp_back(Tensor):
    a = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    b = a.exp()
    b.backward(end_grad=np.array([[1.0, 1.0, 1.0]]))
    assert a.grad is not None
    np.testing.assert_allclose(a.grad.array, 1 / np.e, 0, np.e)
    a = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    b = a.exp()
    c = b.exp()
    c.backward(end_grad=np.array([[1.0, 1.0, 1.0]]))
    def d(x):
        return (np.e**x) * (np.e ** (np.e**x))
    assert a.grad is not None
    np.testing.assert_allclose(a.grad.array, *[d(x) for x in a.array])
    print("All tests in `test_exp_back` passed!")

def test_reshape_back(Tensor):
    a = Tensor([1, 2, 3, 4, 5, 6], requires_grad=True)
    b = a.reshape((3, 2))
    b.backward(end_grad=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]))
    assert a.grad is not None and np.allclose(a.grad.array, np.ones(6))
    print("All tests in `test_reshape_back` passed!")

def test_permute_back(Tensor):
    a = Tensor(np.arange(24).reshape((2, 3, 4)), requires_grad=True)
    out = a.permute((2, 0, 1))
    out.backward(np.arange(24).reshape((4, 2, 3)))
    assert a.grad is not None
    np.testing.assert_allclose(
        a.grad.array,
        np.array([
            [[0.0, 6.0, 12.0, 18.0], [1.0, 7.0, 13.0, 19.0], [2.0, 8.0, 14.0, 20.0]],
            [[3.0, 9.0, 15.0, 21.0], [4.0, 10.0, 16.0, 22.0], [5.0, 11.0, 17.0, 23.0]],
        ]),
    )
    print("All tests in `test_permute_back` passed!")

def test_expand(Tensor):
    a = Tensor(np.ones((2, 1, 3)), requires_grad=True)
    b = a.expand((5, 1, 2, 4, 3))
    b.backward(np.full_like(b.array, 10.0))
    assert a.grad is not None and a.grad.shape == a.array.shape
    assert (a.grad.array == 20 * 10.0).all()
    print("All tests in `test_expand` passed!")

def test_expand_negative_length(Tensor):
    a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    b = a.expand((3, 2, -1))
    assert b.shape == (3, 2, 5)
    b.backward(end_grad=np.ones(b.shape))
    assert a.grad is not None and a.grad.shape == a.array.shape
    assert (a.grad.array == 6).all()
    print("All tests in `test_expand_negative_length` passed!")

def test_sum_keepdim_false(Tensor):
    a = Tensor(np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]), requires_grad=True)
    b = a.sum(0)
    c = b.sum(0)
    c.backward(np.array(2))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 2).all()
    print("All tests in `test_sum_keepdim_false` passed!")

def test_sum_keepdim_true(Tensor):
    a = Tensor(np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]), requires_grad=True)
    b = a.sum(1, keepdim=True)
    c = a.sum(0, keepdim=True)
    np.testing.assert_almost_equal(c.array, np.array([[5.0, 7.0, 9.0, 11.0, 13.0]]))
    c.backward(end_grad=np.ones(c.shape))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 1).all()
    print("All tests in `test_sum_keepdim_true` passed!")

def test_sum_dim_none(Tensor):
    a = Tensor(np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]), requires_grad=True)
    b = a.sum()
    b.backward(np.array(4))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 4).all()
    print("All tests in `test_sum_dim_none` passed!")

def test_getitem_int(Tensor):
    a = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
    b = a[1]
    c = b.sum(0)
    c.backward(np.array(10.0))
    assert a.grad is not None and np.allclose(a.grad.array, np.array([[0, 0, 0], [10, 10, 10]]))
    print("All tests in `test_getitem_int` passed!")

def test_getitem_tuple(Tensor):
    a = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
    b = a[(1, 2)]
    b.backward(np.array(10.0))
    assert a.grad is not None and np.allclose(a.grad.array, np.array([[0, 0, 0], [0, 0, 10]]))
    print("All tests in `test_getitem_tuple` passed!")

def test_getitem_integer_array(Tensor):
    a = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
    index = np.array([0, 1, 0, 1, 0]), np.array([0, 0, 1, 2, 0])
    out = a[index]
    out.sum().backward(np.array(10.0))
    assert a.grad is not None
    np.testing.assert_allclose(a.grad.array, np.array([[20, 10, 0], [10, 0, 10]]))
    print("All tests in `test_getitem_integer_array` passed!")

def test_getitem_integer_tensor(Tensor):
    a = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
    index = Tensor(np.array([0, 1, 0, 1, 0])), Tensor(np.array([0, 0, 1, 2, 0]))
    out = a[index]
    out.sum().backward(np.array(10.0))
    assert a.grad is not None
    np.testing.assert_allclose(a.grad.array, np.array([[20, 10, 0], [10, 0, 10]]))
    print("All tests in `test_getitem_integer_tensor` passed!")

def test_add_broadcasted(Tensor):
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = Tensor([[0], [1], [10]], requires_grad=True)
    c = a + b
    c.backward(end_grad=np.ones(c.shape))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 3).all()
    assert b.grad is not None
    assert b.grad.shape == b.shape
    assert (b.grad.array == 4).all()
    print("All tests in `test_add_broadcasted` passed!")

def test_subtract_broadcasted(Tensor):
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = Tensor([[0], [1], [10]], requires_grad=True)
    c = a - b
    c.backward(end_grad=np.ones(c.shape))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 3).all()
    assert b.grad is not None
    assert b.grad.shape == b.shape
    assert (b.grad.array == -4).all()
    print("All tests in `test_subtract_broadcasted` passed!")

def test_truedivide_broadcasted(Tensor):
    a = Tensor([0, 6, 12, 18], requires_grad=True)
    b = Tensor([[1], [2], [3]], requires_grad=True)
    c = a / b
    c.backward(end_grad=np.ones(c.shape))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == (1 + 1 / 2 + 1 / 3)).all()
    assert b.grad is not None
    assert b.grad.shape == b.shape
    assert np.equal(b.grad.array, np.array([[-36.0], [-9.0], [-4.0]])).all()
    print("All tests in `test_truedivide_broadcasted` passed!")

def test_maximum(Tensor):
    a = Tensor([0, 1, 2], requires_grad=True)
    b = Tensor([-1, 1, 3], requires_grad=True)
    out = a.maximum(b)
    np.testing.assert_allclose(out.array, [0, 1, 3])
    out.backward(end_grad=np.ones(out.shape))
    assert a.grad is not None
    assert b.grad is not None
    np.testing.assert_allclose(a.grad.array, [1, 0.5, 0])
    np.testing.assert_allclose(b.grad.array, [0, 0.5, 1])
    print("All tests in `test_maximum` passed!")

def test_maximum_broadcasted(Tensor):
    a = Tensor([0, 1, 2], requires_grad=True)
    b = Tensor([[-1], [1], [3]], requires_grad=True)
    out = a.maximum(b)
    np.testing.assert_allclose(out.array, np.array([[0, 1, 2], [1, 1, 2], [3, 3, 3]]))
    out.backward(end_grad=np.ones(out.shape))
    assert a.grad is not None and np.allclose(a.grad.array, np.array([1.0, 1.5, 2.0]))
    assert b.grad is not None and np.allclose(b.grad.array, np.array([[0.0], [1.5], [3.0]]))
    print("All tests in `test_maximum_broadcasted` passed!")

def test_relu(Tensor):
    a = Tensor([-1, 0, 1], requires_grad=True)
    out = a.relu()
    out.backward(end_grad=np.ones(out.shape))
    assert a.grad is not None and np.allclose(a.grad.array, np.array([0, 0.5, 1.0]))
    print("All tests in `test_relu` passed!")

def test_matmul2d(Tensor):
    a = Tensor(np.arange(-3, 3).reshape((2, 3)), requires_grad=True)
    b = Tensor(np.arange(-4, 5).reshape((3, 3)), requires_grad=True)
    out = a @ b
    out.backward(end_grad=np.ones(out.shape))
    assert a.grad is not None
    assert b.grad is not None
    np.testing.assert_allclose(a.grad.array, np.array([[-9, 0, 9], [-9, 0, 9]]))
    np.testing.assert_allclose(b.grad.array, np.array([[-3, -3, -3], [-1, -1, -1], [1, 1, 1]]))
    print("All tests in `test_matmul2d` passed!")

def test_cross_entropy(Tensor, cross_entropy):
    logits = Tensor([
        [float("-inf"), float("-inf"), float("-inf"), 0], 
        [1/4, 1/4, 1/4, 1/4], 
        [float("-inf"), 0, 0, 0]
    ])
    true_labels = Tensor([3, 0, 0])
    expected = Tensor([0.0, np.log(4), float("inf")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = cross_entropy(logits, true_labels)
    np.testing.assert_allclose(actual.array, expected.array)
    print("All tests in `test_cross_entropy` passed!")


def test_conv1d_minimal(conv1d_minimal, n_tests=20):
    import numpy as np
    for _ in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 30)#width
        ci = np.random.randint(1, 5)
        co = np.random.randint(1, 5)
        kernel_size = np.random.randint(1, 10)
        x = t.randn((b, ci, h))
        weights = t.randn((co, ci, kernel_size))
        my_output = conv1d_minimal(x, weights)
        torch_output = t.conv1d(x, weights, stride=1, padding=0)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv1d_minimal` passed!")

def test_conv2d_minimal(conv2d_minimal, n_tests=4):
    """
    Compare against torch.conv2d.
    Due to floating point rounding, they can be quite different in float32 but should be nearly identical in float64.
    """
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        w = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        x = t.randn((b, ci, h, w), dtype=t.float64)
        weights = t.randn((co, ci, *kernel_size), dtype=t.float64)
        my_output = conv2d_minimal(x, weights)
        torch_output = t.conv2d(x, weights)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv2d_minimal` passed!")

def test_conv1d(conv1d, n_tests=10):
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        stride = np.random.randint(1, 5)
        padding = np.random.randint(0, 5)
        kernel_size = np.random.randint(1, 10)
        x = t.randn((b, ci, h))
        weights = t.randn((co, ci, kernel_size))
        my_output = conv1d(x, weights, stride=stride, padding=padding)
        torch_output = t.conv1d(x, weights, stride=stride, padding=padding)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv1d` passed!")

def test_pad1d(pad1d):
    """Should work with one channel of width 4."""
    x = t.arange(4).float().view((1, 1, 4))
    actual = pad1d(x, 1, 3, -2.0)
    expected = t.tensor([[[-2.0, 0.0, 1.0, 2.0, 3.0, -2.0, -2.0, -2.0]]])
    t.testing.assert_close(actual, expected)
    print("All tests in `test_pad1d` passed!")


def test_pad1d_multi_channel(pad1d):
    """Should work with two channels of width 2."""
    x = t.arange(4).float().view((1, 2, 2))
    actual = pad1d(x, 0, 2, -3.0)
    expected = t.tensor([[[0.0, 1.0, -3.0, -3.0], [2.0, 3.0, -3.0, -3.0]]])
    t.testing.assert_close(actual, expected)
    print("All tests in `test_pad1d_multi_channel` passed!")

def test_pad2d(pad2d):
    """Should work with one channel of 2x2."""
    x = t.arange(4).float().view((1, 1, 2, 2))
    expected = t.tensor([[[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 3.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]]])
    actual = pad2d(x, 0, 1, 2, 3, 0.0)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_pad2d` passed!")

def test_pad2d_multi_channel(pad2d):
    """Should work with two channels of 2x1."""
    x = t.arange(4).float().view((1, 2, 2, 1))
    expected = t.tensor([[[[-1.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]], [[-1.0, 2.0], [-1.0, 3.0], [-1.0, -1.0]]]])
    actual = pad2d(x, 1, 0, 0, 1, -1.0)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_pad2d_multi_channel` passed!")

def test_conv2d(conv2d, n_tests=5):
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        w = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        stride = tuple(np.random.randint(1, 5, size=(2,)))
        padding = tuple(np.random.randint(0, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        x = t.randn((b, ci, h, w), dtype=t.float64)
        weights = t.randn((co, ci, *kernel_size), dtype=t.float64)
        my_output = conv2d(x, weights, stride=stride, padding=padding)
        torch_output = t.conv2d(x, weights, stride=stride, padding=padding)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv2d` passed!")

def test_maxpool2d(my_maxpool2d, n_tests=20):
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 50)
        w = np.random.randint(10, 50)
        ci = np.random.randint(1, 20)
        stride = None if np.random.random() < 0.5 else tuple(np.random.randint(1, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        kH, kW = kernel_size
        padding = np.random.randint(0, 1 + kH // 2), np.random.randint(0, 1 + kW // 2)
        x = t.randn((b, ci, h, w))
        my_output = my_maxpool2d(
            x,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        torch_output = t.max_pool2d(
            x,
            kernel_size,
            stride=stride,  # type: ignore (None actually is allowed)
            padding=padding,
        )
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_maxpool2d` passed!")

def test_maxpool2d_module(MaxPool2d, n_tests=20):
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 50)
        w = np.random.randint(10, 50)
        ci = np.random.randint(1, 20)
        stride = None if np.random.random() < 0.5 else tuple(np.random.randint(1, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        kH, kW = kernel_size
        padding = np.random.randint(0, 1 + kH // 2), np.random.randint(0, 1 + kW // 2)
        x = t.randn((b, ci, h, w))
        my_output = MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding,
        )(x)

        torch_output = nn.MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding,
        )(x)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_maxpool2d_module` passed!")

def test_conv2d_module(Conv2d, n_tests=5):
    """
    Your weight should be called 'weight' and have an appropriate number of elements.
    """
    m = Conv2d(4, 5, (3, 3))
    assert isinstance(m.weight, t.nn.parameter.Parameter), "Weight should be registered a parameter!"
    assert m.weight.nelement() == 4 * 5 * 3 * 3
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        w = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        stride = tuple(np.random.randint(1, 5, size=(2,)))
        padding = tuple(np.random.randint(0, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        x = t.randn((b, ci, h, w))
        my_conv = Conv2d(in_channels=ci, out_channels=co, kernel_size=kernel_size, stride=stride, padding=padding)
        my_output = my_conv(x)
        torch_output = t.conv2d(x, my_conv.weight, stride=stride, padding=padding)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv2d_module` passed!")

def plot_loss_and_accuracy(loss_list, accuracy_list):

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(y=loss_list, name="Loss"), secondary_y=False)

    n_batches = len(loss_list) // len(accuracy_list)
    x = list(range(n_batches-1, len(loss_list), n_batches))
    fig.add_trace(go.Scatter(y=accuracy_list, x=x, name="Accuracy"), secondary_y=True)

    fig.update_layout(
        title_text="CNN training loss & test accuracy",
        template="simple_white", 
        xaxis_range=[0, len(loss_list)], 
        yaxis2_range=[0, 1],
        yaxis2_tickformat=".0%", 
        hovermode="x unified"
    )

    fig.show()

def test_batchnorm2d_module(BatchNorm2d):
    """The public API of the module should be the same as the real PyTorch version."""
    num_features = 2
    bn = BatchNorm2d(num_features)
    assert bn.num_features == num_features
    assert isinstance(bn.weight, t.nn.parameter.Parameter), f"weight has wrong type: {type(bn.weight)}"
    assert isinstance(bn.bias, t.nn.parameter.Parameter), f"bias has wrong type: {type(bn.bias)}"
    assert isinstance(bn.running_mean, t.Tensor), f"running_mean has wrong type: {type(bn.running_mean)}"
    assert isinstance(bn.running_var, t.Tensor), f"running_var has wrong type: {type(bn.running_var)}"
    assert isinstance(bn.num_batches_tracked, t.Tensor), f"num_batches_tracked has wrong type: {type(bn.num_batches_tracked)}"
    print("All tests in `test_batchnorm2d_module` passed!")

def test_batchnorm2d_forward(BatchNorm2d):
    """For each channel, mean should be very close to 0 and std kinda close to 1 (because of eps)."""
    num_features = 2
    bn = BatchNorm2d(num_features)
    assert bn.training
    x = t.randn((100, num_features, 3, 4))
    out = bn(x)
    assert x.shape == out.shape
    t.testing.assert_close(out.mean(dim=(0, 2, 3)), t.zeros(num_features))
    t.testing.assert_close(out.std(dim=(0, 2, 3)), t.ones(num_features), atol=1e-3, rtol=1e-3)
    print("All tests in `test_batchnorm2d_forward` passed!")

def test_batchnorm2d_running_mean(BatchNorm2d):
    """Over repeated forward calls with the same data in train mode, the running mean should converge to the actual mean."""
    bn = BatchNorm2d(3, momentum=0.6)
    assert bn.training
    x = t.arange(12).float().view((2, 3, 2, 1))
    mean = t.tensor([3.5000, 5.5000, 7.5000])
    num_batches = 30
    for i in range(num_batches):
        bn(x)
        expected_mean = (1 - (((1 - bn.momentum) ** (i + 1)))) * mean
        t.testing.assert_close(bn.running_mean, expected_mean)
    assert bn.num_batches_tracked.item() == num_batches

    # Large enough momentum and num_batches -> running_mean should be very close to actual mean
    bn.eval()
    actual_eval_mean = bn(x).mean((0, 2, 3))
    t.testing.assert_close(actual_eval_mean, t.zeros(3))
    print("All tests in `test_batchnorm2d_running_mean` passed!")

def test_DFT_func(DFT_1d, x=np.linspace(-1, 1), function=lambda x: x**2) -> None:
    
    y = function(x)
    y_DFT = DFT_1d(y)
    y_reconstructed = DFT_1d(y_DFT, inverse=True)
    
    np.testing.assert_allclose(y, y_reconstructed, atol=1e-10)
    
def test_integrate_function(integrate_function):
    
    func = lambda x: np.sin(x) ** 2
    x0 = -np.pi
    x1 = np.pi
    
    integral_approx = integrate_function(func, x0, x1)
    integral_true = 0.5 * (x1 - x0)
    
    np.testing.assert_allclose(integral_true, integral_approx, atol=1e-10)

def test_integrate_product(integrate_product):
    
    func1 = np.sin
    func2 = np.cos 
    x0 = -np.pi
    x1 = np.pi
    
    integrate_product_approx = integrate_product(func1, func2, x0, x1)
    integrate_product_true = 0
    np.testing.assert_allclose(integrate_product_true, integrate_product_approx, atol=1e-10)
    
    integrate_product_approx = integrate_product(func1, func1, x0, x1)
    integrate_product_true = 0.5 * (x1 - x0)
    np.testing.assert_allclose(integrate_product_true, integrate_product_approx, atol=1e-10)

def create_interactive_fourier_graph(calculate_fourier_series: Callable, func: Callable):

    label = wg.Label("Number of terms in Fourier series: ")

    slider = wg.IntSlider(min=0, max=50, value=0)

    x = np.linspace(-np.pi, np.pi, 1000)
    y = func(x)

    fig = go.FigureWidget(
        data = [
            go.Scatter(x=x, y=y, name="Original function", mode="lines"),
            go.Scatter(x=x, y=y, name="Reconstructed function", mode="lines")
        ],
        layout = go.Layout(title_text=r"Original vs reconstructed", template="simple_white", margin_t=100)
    )

    def respond_to_slider(change):
        max_freq = slider.value
        coeffs, func_approx = calculate_fourier_series(func, max_freq)
        fig.data[1].y = np.vectorize(func_approx)(x)

    slider.observe(respond_to_slider)

    respond_to_slider("unimportant text to trigger first response")

    box_layout = wg.Layout(border="solid 1px black", padding="20px", margin="20px", width="80%")

    return wg.VBox([wg.HBox([label, slider], layout=box_layout), fig])

TARGET_FUNC = np.sin
NUM_FREQUENCIES = 4
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6
    
def get_title_from_coeffs(a_0, A_n, B_n):
    A_n_coeffs = " + ".join([f"{a_n:.2f}" + r"\cos{" + (str(n) if n>1 else "") + " x}" for (n, a_n) in enumerate(A_n, 1)])
    B_n_coeffs = " + ".join([f"{b_n:.2f}" + r"\sin{" + (str(n) if n>1 else "") + " x}" for (n, b_n) in enumerate(B_n, 1)])
    return f"{a_0:.2f}" + " + " + A_n_coeffs + " + " + B_n_coeffs + r"$"

def visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list):

    label = wg.Label("Number of steps: ")

    slider = wg.IntSlider(min=0, max=TOTAL_STEPS-100, value=0, step=100)

    fig = go.FigureWidget(
        data = [go.Scatter(x=x, y=y, mode="lines", marker_color="blue")] + [
            go.Scatter(x=x, y=y_pred_list[i], mode="lines", marker_color="rgba(100, 100, 100, 0.1)")
            for i in range(len(y_pred_list))
        ],
        layout = go.Layout(title_text=r"Original vs reconstructed", template="simple_white", margin_t=100, showlegend=False)
    )

    def respond_to_slider(change):
        idx = slider.value // 100
        with fig.batch_update():
            fig.update_layout(title_text = get_title_from_coeffs(*coeffs_list[idx]))
            for i in range(len(list(fig.data))-1):
                fig.data[i+1]["marker"]["color"] = "red" if i == idx else "rgba(100, 100, 100, 0.1)"

    slider.observe(respond_to_slider)

    respond_to_slider("unimportant text to trigger first response")

    box_layout = wg.Layout(border="solid 1px black", padding="20px", margin="20px", width="80%")

    return wg.VBox([wg.HBox([label, slider], layout=box_layout), fig])
