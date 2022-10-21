import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

from megengine.optimizer import Optimizer
from megengine.optimizer.lr_scheduler import LRScheduler

from ..types import number_type

__all__ = [
    'CosineAnnealingLR',
    'CyclicLR',
    'LambdLR',
    'OneCycleLR',
]


def _check_valid(num: Union[number_type, List[number_type]], check_func: Callable):
    if isinstance(num, (float, int)):
        num = [num]
    for i in num:
        assert check_func(i)


class BaseLRScheduler(LRScheduler):
    def state_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != 'optimizer' and key != 'base_lrs'
        }

    def load_state_dict(self, state_dict):
        tmp_dict = {}
        for key in ["T_max", "eta_min", "current_epoch"]:
            if not key in state_dict.keys():
                raise KeyError(
                    "key '{}'' is not specified in "
                    "state_dict when loading state dict".format(key)
                )
            tmp_dict[key] = state_dict[key]

        self.__dict__.update(tmp_dict)


class CosineAnnealingLR(BaseLRScheduler):
    """
    A Simple Implement Of CosineAnnealingLR (https://arxiv.org/abs/1608.03983)
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        current_epoch (int): the index of current epoch. Default: -1.
    """

    def __init__(self, optimizer, T_max, eta_min, current_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, current_epoch)

    def get_lr(self):
        if self.current_epoch == -1:
            return self.base_lrs
        elif (self.current_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.cos(math.pi * self.current_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.current_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]


class LambdLR(BaseLRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Union[Callable, Sequence[Callable]],
        current_epoch: int = -1,
    ):
        super().__init__(optimizer, current_epoch)
        if not isinstance(lr_lambda, Sequence):
            lr_lambda = [lr_lambda] * len(self.base_lrs)
        self.lr_lambda = lr_lambda

    def get_lr(self):
        return [
            base_lr * lamb(self.current_epoch) for base_lr, lamb in zip(self.base_lrs, self.lr_lambda)
        ]


class OneCycleLR(BaseLRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_learning_rate: number_type,
        total_steps: int,
        divide_factor: float = 25.,
        end_learning_rate: number_type = 0.0001,
        phase_pct: float = 0.3,
        anneal_strategy: str = 'cos',
        three_phase: bool = False,
        current_epoch=-1,
    ):

        _check_valid([max_learning_rate, total_steps,
                     divide_factor, end_learning_rate], lambda x: x > 0)

        _check_valid(phase_pct, lambda x: x > 0 and x < 1)

        self.total_steps = total_steps
        self.max_lr = float(max_learning_rate)
        min_lr = float(end_learning_rate)

        initial_lr = max_learning_rate / float(divide_factor)
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
                self._step_config[3] -
                self._step_config[2],  # for the last step.
            ]
            # start lr and end lr of each phase.
            self._lr_config = [
                initial_lr, max_learning_rate, initial_lr, min_lr
            ]
        else:
            self._step_config = [
                0, phase_pct * self.total_steps - 1, self.total_steps - 1,
                self.total_steps - 1
            ]
            self._steps_size = [
                self._step_config[1] - self._step_config[0],
                self._step_config[2] - self._step_config[1],
                self._step_config[2] - self._step_config[1],
            ]
            self._lr_config = [initial_lr, max_learning_rate, min_lr]

        if anneal_strategy == 'cos':
            self.anneal_func = self._cos_annealing
        elif anneal_strategy == 'linear':
            self.anneal_func = self._linear_annealing
        else:
            raise ValueError(
                "'anneal_strategy' must by one of 'cos' or 'linear', but received {}".
                format(anneal_strategy))
        super().__init__(optimizer, current_epoch)

    def _cos_annealing(self, start_lr, end_lr, pct):
        cos_out = math.cos(math.pi * pct) + 1
        return end_lr + (start_lr - end_lr) / 2.0 * cos_out

    def _linear_annealing(self, start_lr, end_lr, pct):
        return (end_lr - start_lr) * pct + start_lr

    def get_lr(self):
        current_step = self.current_epoch

        if current_step > self.total_steps:
            raise ValueError(
                "Tried to step {} times. However the number of total steps is {}"
                .format(current_step, self.total_steps))

        for (i, (end_step, step_size)
             ) in enumerate(zip(self._step_config[1:], self._steps_size)):
            # i == len(self._lr_config) - 2 catch the last step, otherwise it will return None.
            if current_step <= end_step or i == len(self._lr_config) - 2:
                # self._step_config[i] means start step of a phase.
                percentage = (current_step - self._step_config[i]) / step_size
                return [self.anneal_func(self._lr_config[i],
                                         self._lr_config[i + 1], percentage)]


class CyclicLR(BaseLRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        base_learning_rate: number_type,
        max_learning_rate: number_type,
        step_size_up: int,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        exp_gamma: number_type = 1.,
        scale_fn: Optional[Callable] = None,
        scale_mode: str = 'cycle',
        current_epoch: int = -1,
    ):
        _check_valid([base_learning_rate, max_learning_rate,
                     exp_gamma, step_size_up], lambda x: x > 0)

        # do not use base_lrs
        self.base_lr = float(base_learning_rate)
        self.max_lr = float(max_learning_rate)

        self.step_size_up = step_size_up
        self.step_size_down = step_size_up if step_size_down is None else step_size_down

        self.cycle_size = step_size_up + self.step_size_down
        self.step_up_pct = step_size_up / self.cycle_size
        self.amplitude = self.max_lr - base_learning_rate

        if mode not in ['triangular', 'triangular2', 'exp_range'
                        ] and scale_fn is None:
            raise ValueError(
                "'mode' is invalid and 'scale_fn' is not specified, make sure one of 'mode' or 'scale_fn' is valid"
            )
        if scale_mode not in ['cycle', 'iterations']:
            raise ValueError(
                "'scale_mode' must be one of 'cycle' or 'iterations")

        self.mode = mode
        self.gamma = float(exp_gamma)  # only for exp_range mode

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        super().__init__(optimizer, current_epoch)

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2.**(x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**x

    def get_lr(self):
        iterations = self.current_epoch

        cycle = 1 + iterations // self.cycle_size
        pct_per_cycle = 1. + iterations / self.cycle_size - cycle

        if pct_per_cycle <= self.step_up_pct:
            scale_factor = pct_per_cycle / self.step_up_pct
        else:
            scale_factor = (1 - pct_per_cycle) / (1 - self.step_up_pct)

        base_height = self.amplitude * scale_factor

        lr = self.base_lr + base_height * self.scale_fn(eval(self.scale_mode))

        return [lr] * len(self.optimizer.param_groups)
