import megengine as mge
import megengine.module as M
import numpy as np
import torch
from megengine import autodiff
from torch.nn import init
from torch.optim.optimizer import Optimizer
from utils import _init_weights, _test_modules

from megbox.optimizer import Lion, Tiger

GLOBAL_RTOL = 1e-3
GLOBAL_ATOL = 1e-5


class TorchLion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.mul_(1 - group["lr"] * group["weight_decay"])
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


class TorchTiger(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.965, weight_decay=0.01):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.mul_(1 - group["lr"] * group["weight_decay"])
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                beta = group["beta"]
                update = beta * exp_avg + (1 - beta) * grad
                p.add_(torch.sign(update), alpha=-group["lr"])
        return loss


def _torch_init_weights(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        init.ones_(m.weight)
        if m.bias is not None:
            init.ones_(m.bias)


class TorchTestNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=True)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.mlp = torch.nn.Linear(5, 1)
        self.apply(_torch_init_weights)

    def forward(self, x):
        x = self.gap(self.conv(x)).squeeze(-1).squeeze(-1)
        return self.mlp(x)


class TestNet(M.Module):
    def __init__(self):
        super().__init__()
        self.conv = M.Conv2d(3, 5, 3, bias=True)
        self.gap = M.AdaptiveAvgPool2d(1)
        self.mlp = M.Linear(5, 1)
        self.apply(_init_weights)

    def forward(self, x):
        x = mge.functional.squeeze(self.gap(self.conv(x)), [-1, -2])
        return self.mlp(x)


OPTIM_KWARGS = dict(
    lion=[
        dict(lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0),
        dict(lr=1e-4, betas=(0.9, 0.99), weight_decay=0.001),
    ],
    tiger=[
        dict(lr=1e-4, weight_decay=0.0),
        dict(lr=1e-4, weight_decay=0.001),
    ],
)

OPTIMIZER = dict(lion=Lion, tiger=Tiger)
TORCH_OPTIM = dict(lion=TorchLion, tiger=TorchTiger)


def get_name(value):
    for k, v in OPTIMIZER.items():
        if value == v:
            return k
    raise ValueError


def get_grad(model):
    return dict(
        conv_w=model.conv.weight.grad,
        conv_b=model.conv.bias.grad,
        mlp=model.conv.weight.grad,
    )


def test_optim():
    spatial_sizes = [1]

    def check_func(cls, kwargs, sp_size, name):
        cls_name = get_name(cls)
        torch_cls = TORCH_OPTIM[cls_name]
        torch_model = TorchTestNet()
        if torch.cuda.is_available():
            torch_model.cuda()
        torch_module = torch_cls(torch_model.parameters(), **kwargs)

        model = TestNet()
        model.train()
        module = cls(model.parameters(), **kwargs)
        gm = autodiff.GradManager().attach(model.parameters())

        grad_rec = []
        loss_rec = []
        torch_grad_rec = []
        torch_loss_rec = []
        for _ in range(10):
            with gm:
                x = mge.tensor(np.ones((2, 3, 9, 9))).astype("float32")
                y = model(x)
                loss = y.sum()
                loss_rec.append(loss.numpy())
                gm.backward(loss)
                module.step()
                grad_rec.append(get_grad(model))
                module.clear_grad()

        for _ in range(10):
            x = torch.tensor(np.ones((2, 3, 9, 9)), dtype=torch.float32)
            if torch.cuda.is_available():
                x = x.cuda()
            y = torch_model(x)
            loss = y.sum()
            loss.backward()
            if torch.cuda.is_available():
                loss = loss.cpu()
            torch_loss_rec.append(loss.detach().numpy())
            torch_module.step()
            torch_grad_rec.append(get_grad(torch_model))
            torch_module.zero_grad()

        for m, t in zip(grad_rec, torch_grad_rec):
            for k in ["conv_w", "conv_b", "mlp"]:
                md = m[k].numpy()
                td = m[k]
                if torch.cuda.is_available():
                    td = td.cpu()
                td = td.detach().numpy()
                np.testing.assert_allclose(md, td, atol=GLOBAL_ATOL, rtol=GLOBAL_RTOL)

        for m, t in zip(loss_rec, torch_loss_rec):
            np.testing.assert_allclose(m, t, atol=GLOBAL_ATOL, rtol=GLOBAL_RTOL)

        mp = {k: v for k, v in model.named_parameters()}
        tp = {k: v for k, v in torch_model.named_parameters()}

        for mn, p1 in mp.items():
            p1 = p1.numpy()
            p2 = tp[mn]
            if torch.cuda.is_available():
                p2 = p2.cpu()
            p2 = p2.detach().numpy()
            if mn == "conv.bias":
                p2 = np.expand_dims(p2, (0, 2, 3))
            np.testing.assert_allclose(p1, p2, atol=GLOBAL_ATOL, rtol=GLOBAL_RTOL)

    _test_modules(
        module_mapper=OPTIMIZER,
        kwargs_mapper=OPTIM_KWARGS,
        spatial_sizes=spatial_sizes,
        check_func=check_func,
    )


if __name__ == "__main__":
    test_optim()
