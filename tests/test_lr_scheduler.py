import megengine as mge
import torch
from megengine.optimizer import SGD
from torch.optim import SGD as t_SGD
from torch.optim.lr_scheduler import CosineAnnealingLR as t_CosineAnnealingLR
from torch.optim.lr_scheduler import CyclicLR as t_CyclicLR
from torch.optim.lr_scheduler import LambdaLR as t_LambdaLR
from torch.optim.lr_scheduler import OneCycleLR as t_OneCycleLR

from megbox.optimizer.lr_scheduler import (CosineAnnealingLR, CyclicLR,
                                           LambdLR, OneCycleLR)
from megbox.optimizer.params_helper import filter_params

from .utils import _test_modules

LRSCHEDULERS = dict(
    cos=CosineAnnealingLR,
    cyclic=CyclicLR,
    lambd=LambdLR,
    onecycle=OneCycleLR,
)

T_LRSCHEDULERS = dict(
    cos=t_CosineAnnealingLR,
    cyclic=t_CyclicLR,
    lambd=t_LambdaLR,
    onecycle=t_OneCycleLR,
)


def func1(x):
    return x


def func2(x):
    return x / 2 + 0.5


MGEKWARGS = dict(
    cos=[dict(T_max=200, eta_min=0.1), dict(T_max=10, eta_min=0.2)],
    cyclic=[
        dict(max_lr=1, step_size_up=5, mode="triangular"),
        dict(max_lr=2, step_size_up=6, mode="triangular2"),
        dict(max_lr=0.9, step_size_up=8, mode="exp_range"),
        dict(max_lr=0.9, step_size_up=5, mode="exp_range", gamma=0.99),
    ],
    lambd=[dict(lr_lambda=func1), dict(lr_lambda=func2)],
    onecycle=[
        dict(max_lr=0.3, total_steps=12, divide_factor=3),
        dict(max_lr=0.3, total_steps=15, anneal_strategy="cos", divide_factor=3),
        dict(
            max_lr=0.3,
            total_steps=15,
            anneal_strategy="cos",
            three_phase=True,
            divide_factor=3,
        ),
        dict(
            max_lr=0.3,
            total_steps=15,
            anneal_strategy="linear",
            three_phase=True,
            divide_factor=3,
        ),
        dict(
            max_lr=0.3,
            total_steps=15,
            anneal_strategy="cos",
            three_phase=False,
            divide_factor=3,
        ),
        dict(
            max_lr=0.3,
            total_steps=15,
            anneal_strategy="linear",
            three_phase=False,
            divide_factor=3,
        ),
    ],
)

TKWARGS = dict(
    cyclic=[
        dict(base_lr=0.1, max_lr=1, step_size_up=5, mode="triangular"),
        dict(base_lr=0.1, max_lr=2, step_size_up=6, mode="triangular2"),
        dict(base_lr=0.1, max_lr=0.9, step_size_up=8, mode="exp_range"),
        dict(base_lr=0.1, max_lr=0.9, step_size_up=5, mode="exp_range", gamma=0.99),
    ],
    onecycle=[
        dict(max_lr=0.3, total_steps=12, final_div_factor=1e3, div_factor=3),
        dict(
            max_lr=0.3,
            total_steps=15,
            anneal_strategy="cos",
            final_div_factor=1e3,
            div_factor=3,
        ),
        dict(
            max_lr=0.3,
            total_steps=15,
            anneal_strategy="cos",
            three_phase=True,
            final_div_factor=1e3,
            div_factor=3,
        ),
        dict(
            max_lr=0.3,
            total_steps=15,
            anneal_strategy="linear",
            three_phase=True,
            final_div_factor=1e3,
            div_factor=3,
        ),
        dict(
            max_lr=0.3,
            total_steps=15,
            anneal_strategy="cos",
            three_phase=False,
            final_div_factor=1e3,
            div_factor=3,
        ),
        dict(
            max_lr=0.3,
            total_steps=15,
            anneal_strategy="linear",
            three_phase=False,
            final_div_factor=1e3,
            div_factor=3,
        ),
    ],
)

MULTI_MGEKWARGS = dict(
    cos=[dict(T_max=200, eta_min=0.1), dict(T_max=10, eta_min=0.2)],
    cyclic=[
        dict(max_lr=[1, 2], step_size_up=5, mode="triangular"),
        dict(max_lr=[2, 1], step_size_up=6, mode="triangular2"),
        dict(max_lr=[0.9, 0.7], step_size_up=8, mode="exp_range"),
        dict(max_lr=[0.4, 0.9], step_size_up=5, mode="exp_range", gamma=0.99),
    ],
    lambd=[dict(lr_lambda=[func1, func2])],
    onecycle=[
        dict(
            max_lr=[0.3, 0.3], total_steps=12, divide_factor=3, end_lr=[0.0001, 0.0001]
        ),
        dict(
            max_lr=[0.3, 0.6],
            total_steps=12,
            anneal_strategy="cos",
            divide_factor=3,
            end_lr=[0.0001, 0.0002],
        ),
        dict(
            max_lr=[0.3, 0.6],
            total_steps=12,
            anneal_strategy="cos",
            three_phase=True,
            divide_factor=3,
            end_lr=[0.0001, 0.0002],
        ),
        dict(
            max_lr=[0.3, 0.6],
            total_steps=12,
            anneal_strategy="linear",
            three_phase=True,
            divide_factor=3,
            end_lr=[0.0001, 0.0002],
        ),
        dict(
            max_lr=[0.3, 0.6],
            total_steps=12,
            anneal_strategy="cos",
            three_phase=False,
            divide_factor=3,
            end_lr=[0.0001, 0.0002],
        ),
        dict(
            max_lr=[0.3, 0.6],
            total_steps=12,
            anneal_strategy="linear",
            three_phase=False,
            divide_factor=3,
            end_lr=[0.0001, 0.0002],
        ),
    ],
)
MULTI_TKWARGS = dict(
    cyclic=[
        dict(base_lr=[0.05, 0.1], max_lr=[1, 2], step_size_up=5, mode="triangular"),
        dict(base_lr=[0.05, 0.1], max_lr=[2, 1], step_size_up=6, mode="triangular2"),
        dict(base_lr=[0.05, 0.1], max_lr=[0.9, 0.7], step_size_up=8, mode="exp_range"),
        dict(
            base_lr=[0.05, 0.1],
            max_lr=[0.4, 0.9],
            step_size_up=5,
            mode="exp_range",
            gamma=0.99,
        ),
    ],
    onecycle=[
        dict(max_lr=[0.3, 0.3], total_steps=12, final_div_factor=1e3, div_factor=3),
        dict(
            max_lr=[0.3, 0.6],
            total_steps=12,
            anneal_strategy="cos",
            final_div_factor=1e3,
            div_factor=3,
        ),
        dict(
            max_lr=[0.3, 0.6],
            total_steps=12,
            anneal_strategy="cos",
            three_phase=True,
            final_div_factor=1e3,
            div_factor=3,
        ),
        dict(
            max_lr=[0.3, 0.6],
            total_steps=12,
            anneal_strategy="linear",
            three_phase=True,
            final_div_factor=1e3,
            div_factor=3,
        ),
        dict(
            max_lr=[0.3, 0.6],
            total_steps=12,
            anneal_strategy="cos",
            three_phase=False,
            final_div_factor=1e3,
            div_factor=3,
        ),
        dict(
            max_lr=[0.3, 0.6],
            total_steps=12,
            anneal_strategy="linear",
            three_phase=False,
            final_div_factor=1e3,
            div_factor=3,
        ),
    ],
)


class TorchNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, bias=True)


class MgeNet(mge.module.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = mge.module.Conv2d(3, 3, 3, bias=True)

    def forward(self, x):
        pass


def test_lr_scheduler(with_parma_groups=False):
    sp = [1]

    def check_func(cls, kwargs, sp_size, name, is_gpu):
        m_net = MgeNet()
        t_net = TorchNet()
        if with_parma_groups:
            m_p = filter_params(m_net, [dict(name="weight", lr=0.05)])
            t_p = filter_params(t_net, [dict(name="weight", lr=0.05)])
        else:
            m_p = m_net.parameters()
            t_p = t_net.parameters()
        m_op = SGD(m_p, lr=0.1)
        t_op = t_SGD(t_p, lr=0.1)
        m_cls, t_cls = cls
        m_kwargs, t_kwargs = kwargs
        m_kwargs["optimizer"] = m_op
        t_kwargs["optimizer"] = t_op
        m_lr = m_cls(**m_kwargs)
        t_lr = t_cls(**t_kwargs)
        error_msg = (
            f"Different output of module {name}, with kwargs {kwargs}"  # noqa: E501
        )
        for _ in range(10):
            lr1 = m_lr.get_last_lr()
            lr2 = t_lr.get_last_lr()

            for m, t in zip(lr1, lr2):
                assert m == t, error_msg
            m_lr.step()
            t_lr.step()

    _test_modules(
        module_mappers=[LRSCHEDULERS, T_LRSCHEDULERS],
        kwargs_mappers=[MULTI_MGEKWARGS, MULTI_TKWARGS]
        if with_parma_groups
        else [MGEKWARGS, TKWARGS],
        spatial_sizes=sp,
        check_func=check_func,
    )


if __name__ == "__main__":
    test_lr_scheduler()
    test_lr_scheduler(True)
