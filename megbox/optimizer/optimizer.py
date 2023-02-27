from typing import Tuple

import megengine.functional as F
from megengine import Tensor
# from megengine.functional.inplace import _inplace_add_
from megengine.optimizer import Optimizer


class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        assert all([0.0 <= beta <= 1.0 for beta in betas])

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
            param.set_value(param * (1 - _lr * _weight_decay))

            # Weight update
            update = exp_avg * _beta0 + grad * (1 - _beta0)

            # don't support `Parameter` input
            # _inplace_add_(param, F.sign(update), alpha=make_scalar(1), beta=-_lr)
            param.set_value(param + F.sign(update) * -_lr)

            # Decay the momentum running average coefficient
            exp_avg.set_value(exp_avg * _beta1)

            # _inplace_add_(exp_avg, grad, alpha=1, beta=1 - _beta1)
            exp_avg.set_value(exp_avg + (1 - _beta1) * grad)