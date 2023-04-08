from typing import Tuple

import megengine.functional as F
from megengine import Tensor
# from megengine.functional.inplace import _inplace_add_
from megengine.optimizer import Optimizer


class Lion(Optimizer):  # pylint: disable=abstract-method
    def __init__(
        self,
        params,
        lr: float,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        for beta in betas:
            if beta < 0 or beta > 1:
                raise ValueError(
                    "`beta` must be between [0, 1] but got {}".format(betas)
                )

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)

        super().__init__(params, defaults)

    def _create_state(self, param_group):
        for param in param_group["params"]:
            self._add_state(param, "exp_avg")

    def _updates(self, param_group):
        lr = param_group["lr"]
        weight_decay = param_group["weight_decay"]
        beta0, beta1 = param_group["betas"]

        def make_scalar(val):
            return Tensor(val, dtype="float32")

        _weight_decay = make_scalar(weight_decay)
        _lr, _beta0, _beta1 = map(make_scalar, (lr, beta0, beta1))

        for param in param_group["params"]:
            if param.grad is None:
                continue
            grad = param.grad

            states = self._state[param]

            exp_avg = states["exp_avg"]

            # Perform stepweight decay
            param[...] = param * (1 - _lr * _weight_decay)

            # Weight update
            update = exp_avg * _beta0 + grad * (1 - _beta0)

            # don't support `Parameter` input
            # _inplace_add_(param, F.sign(update), alpha=make_scalar(1), beta=-_lr)
            param[...] = param + F.sign(update) * -_lr

            # Decay the momentum running average coefficient
            exp_avg[...] = exp_avg * _beta1

            # _inplace_add_(exp_avg, grad, alpha=1, beta=1 - _beta1)
            exp_avg[...] = exp_avg + (1 - _beta1) * grad


class Tiger(Optimizer):  # pylint: disable=abstract-method
    """
    Tight-fisted Optimizer(Tiger).
    Reference: https://github.com/bojone/tiger
    """

    def __init__(
        self,
        params,
        lr: float,
        beta: float = 0.965,
        weight_decay: float = 0.01,
    ):
        if lr <= 0.0:
            raise ValueError("`lr` must be great than 0, but got {}".format(lr))
        if beta < 0 or beta > 1:
            raise ValueError("`beta` must be between [0, 1] but got {}".format(beta))

        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)

        super().__init__(params, defaults)

    def _create_state(self, param_group):
        for param in param_group["params"]:
            self._add_state(param, "exp_avg")

    def _updates(self, param_group):
        lr = param_group["lr"]
        weight_decay = param_group["weight_decay"]
        beta = param_group["beta"]

        def make_scalar(val):
            return Tensor(val, dtype="float32")

        _weight_decay = make_scalar(weight_decay)
        _lr, _beta = map(make_scalar, (lr, beta))

        for param in param_group["params"]:
            if param.grad is None:
                continue
            grad = param.grad

            states = self._state[param]

            exp_avg = states["exp_avg"]

            # Perform stepweight decay
            param[...] = param * (1 - _lr * _weight_decay)

            # Weight update
            update = exp_avg * _beta + grad * (1 - _beta)

            param[...] = param + F.sign(update) * -_lr
