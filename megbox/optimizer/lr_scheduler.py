import math
from typing import Callable, List, Optional, Sequence, Union

from megengine.optimizer import Optimizer

from ..types import number_type

__all__ = [
    "CosineAnnealingLR",
    "CyclicLR",
    "LambdLR",
    "OneCycleLR",
]


def _check_valid(num: Union[number_type, List[number_type]], check_func: Callable):
    if isinstance(num, (float, int)):
        num = [num]
    for i in num:
        assert check_func(i)


class LRScheduler:
    __IGNORE_KEYS__ = ["optimizer"]

    def __init__(  # pylint: disable=too-many-branches
        self, optimizer: Optimizer, current_epoch: int = -1
    ):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                "optimizer argument given to the lr_scheduler should be Optimizer"
            )
        self.optimizer = optimizer
        self.current_epoch = current_epoch
        if current_epoch == -1:
            for group in self.optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified in "
                        "param_groups[{}] when resuming an optimizer".format(i)
                    )
        self.base_lrs = [group["initial_lr"] for group in self.optimizer.param_groups]
        self._last_lr = None
        self.ignore_keys = LRScheduler.__IGNORE_KEYS__ + self.__IGNORE_KEYS__

        self.step()

    def state_dict(self):
        r"""Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in self.ignore_keys
        }

    def load_state_dict(self, state_dict):
        r"""Loads the schedulers state.

        Args:
            state_dict: scheduler state.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        r"""Return last computed learing rate"""
        return self._last_lr

    def get_lr(self):
        r"""Compute current learning rate for the scheduler."""
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            self.current_epoch += 1
        else:
            self.current_epoch = epoch

        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group["lr"] = lr


class CosineAnnealingLR(LRScheduler):
    """
    CosineAnnealingLR (https://arxiv.org/abs/1608.03983)
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        current_epoch (int): the index of current epoch. Default: -1.
    """

    def __init__(self, optimizer, T_max, eta_min, current_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, current_epoch)

    def get_lr(self):
        if self.current_epoch == -1:
            return self.base_lrs
        elif (self.current_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * self.current_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.current_epoch - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]


class LambdLR(LRScheduler):
    __IGNORE_KEYS__ = ["lr_lambda"]
    """
    Set learing rate by given functions.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (Callable | List[Callable]): Calculating functions.
        current_epoch (int): the index of current epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Union[Callable, Sequence[Callable]],
        current_epoch: int = -1,
    ):
        super().__init__(optimizer, current_epoch)
        if not isinstance(lr_lambda, Sequence):
            lr_lambda = [lr_lambda] * len(self.base_lrs)
        elif len(lr_lambda) != len(optimizer.param_groups):
            raise ValueError(
                "Expected `lr_lambda` to have the same length of \
                `param_groups`, but got {}".format(
                    len(lr_lambda)
                )
            )
        self.lr_lambda = lr_lambda

    def get_lr(self):
        return [
            base_lr * lamb(self.current_epoch)
            for base_lr, lamb in zip(self.base_lrs, self.lr_lambda)
        ]


class OneCycleLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: number_type,
        total_steps: int,
        divide_factor: float = 25.0,
        end_lr: number_type = 0.0001,
        phase_pct: float = 0.3,
        anneal_strategy: str = "cos",
        three_phase: bool = False,
        current_epoch=-1,
    ):

        _check_valid(
            [total_steps, divide_factor],
            lambda x: x > 0,
        )
        if isinstance(max_lr, (int, float)):
            max_lr = [max_lr]
        elif isinstance(max_lr, (list, tuple)):
            if len(max_lr) != len(self.base_lrs):
                raise ValueError()

        if isinstance(end_lr, (int, float)):
            end_lr = [end_lr]
        elif isinstance(end_lr, (list, tuple)):
            if len(end_lr) != len(self.base_lrs):
                raise ValueError()
            end_lr = [float(i) for i in end_lr]

        _check_valid(phase_pct, lambda x: 1 > x > 0)

        self.total_steps = total_steps
        self.max_lr = max_lr
        min_lr = end_lr

        initial_lr = max_lr / float(divide_factor)
        if three_phase:
            # Note: not sure for now
            if phase_pct >= 0.5:
                raise ValueError(
                    "When three_phase is True, 'phase_pct' must be less than 0.5"
                )
            # start step and end step of each phase.
            self._step_config = [
                0,
                phase_pct * self.total_steps - 1,
                2 * phase_pct * self.total_steps - 2,
                self.total_steps - 1,
                self.total_steps - 1,  # for the last step.
            ]
            # step size of each phase.
            self._steps_size = [
                self._step_config[1] - self._step_config[0],
                self._step_config[2] - self._step_config[1],
                self._step_config[3] - self._step_config[2],
                self._step_config[3] - self._step_config[2],  # for the last step.
            ]
            # start lr and end lr of each phase.
            self._lr_config = [initial_lr, max_lr, initial_lr, min_lr]
        else:
            self._step_config = [
                0,
                phase_pct * self.total_steps - 1,
                self.total_steps - 1,
                self.total_steps - 1,
            ]
            self._steps_size = [
                self._step_config[1] - self._step_config[0],
                self._step_config[2] - self._step_config[1],
                self._step_config[2] - self._step_config[1],
            ]
            self._lr_config = [initial_lr, max_lr, min_lr]

        if anneal_strategy == "cos":
            self.anneal_func = self._cos_annealing
        elif anneal_strategy == "linear":
            self.anneal_func = self._linear_annealing
        else:
            raise ValueError(
                "'anneal_strategy' must by one of 'cos' or 'linear', but received {}".format(
                    anneal_strategy
                )
            )
        super().__init__(optimizer, current_epoch)

    def _cos_annealing(self, start_lr, end_lr, pct):
        cos_out = math.cos(math.pi * pct) + 1
        return end_lr + (start_lr - end_lr) / 2.0 * cos_out

    def _linear_annealing(self, start_lr, end_lr, pct):
        return (end_lr - start_lr) * pct + start_lr

    def get_lr(self):  # pylint: disable=inconsistent-return-statements
        current_step = self.current_epoch

        if current_step > self.total_steps:
            raise ValueError(
                "Tried to step {} times. However the number of total steps is {}".format(
                    current_step, self.total_steps
                )
            )

        for (i, (end_step, step_size)) in enumerate(
            zip(self._step_config[1:], self._steps_size)
        ):
            # i == len(self._lr_config) - 2 catch the last step
            # otherwise it will return None.
            if current_step <= end_step or i == len(self._lr_config) - 2:
                # self._step_config[i] means start step of a phase.
                percentage = (current_step - self._step_config[i]) / step_size
                return [
                    self.anneal_func(
                        self._lr_config[i], self._lr_config[i + 1], percentage
                    )
                ]


class CyclicLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        base_learning_rate: number_type,
        max_learning_rate: number_type,
        step_size_up: int,
        step_size_down: Optional[int] = None,
        mode: str = "triangular",
        exp_gamma: number_type = 1.0,
        scale_fn: Optional[Callable] = None,
        scale_mode: str = "cycle",
        current_epoch: int = -1,
    ):
        _check_valid(
            [base_learning_rate, max_learning_rate, exp_gamma, step_size_up],
            lambda x: x > 0,
        )

        # do not use base_lrs
        self.base_lr = float(base_learning_rate)
        self.max_lr = float(max_learning_rate)

        self.step_size_up = step_size_up
        self.step_size_down = step_size_up if step_size_down is None else step_size_down

        self.cycle_size = step_size_up + self.step_size_down
        self.step_up_pct = step_size_up / self.cycle_size
        self.amplitude = self.max_lr - base_learning_rate

        if mode not in ["triangular", "triangular2", "exp_range"] and scale_fn is None:
            raise ValueError(
                """'mode' is invalid and 'scale_fn' is not specified,
                make sure one of 'mode' or 'scale_fn' is valid"""
            )
        if scale_mode not in ["cycle", "iterations"]:
            raise ValueError("'scale_mode' must be one of 'cycle' or 'iterations")

        self.mode = mode
        self.gamma = float(exp_gamma)  # only for exp_range mode

        if scale_fn is None:
            if self.mode == "triangular":
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = "cycle"
            elif self.mode == "triangular2":
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = "cycle"
            elif self.mode == "exp_range":
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = "iterations"
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        super().__init__(optimizer, current_epoch)

    def _triangular_scale_fn(self, _):
        return 1.0

    def _triangular2_scale_fn(self, x):
        return 1 / (2.0 ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**x

    def get_lr(self):
        iterations = self.current_epoch

        cycle = 1 + iterations // self.cycle_size
        pct_per_cycle = 1.0 + iterations / self.cycle_size - cycle

        if pct_per_cycle <= self.step_up_pct:
            scale_factor = pct_per_cycle / self.step_up_pct
        else:
            scale_factor = (1 - pct_per_cycle) / (1 - self.step_up_pct)

        base_height = self.amplitude * scale_factor

        if self.scale_mode == "cycle":
            scale = self.scale_fn(cycle)
        else:
            scale = self.scale_fn(iterations)

        return self.base_lr * base_height * scale
        # return [lr * base_height * scale  for lr in self.base_lr]
