import math
from typing import Callable, List, Optional, Sequence, Union

from megengine.optimizer import Optimizer

from ..types import Number

__all__ = [
    "CosineAnnealingLR",
    "CyclicLR",
    "LambdLR",
    "OneCycleLR",
]


def _check_valid(
    num: Union[Number, List[Number]], check_func: Callable, error_msg: str = ""
):
    if isinstance(num, (float, int)):
        num = [num]
    for i in num:
        assert check_func(i), error_msg


def _check_lrs(lr, name, expected_length):
    if isinstance(lr, (int, float)):
        lr = [lr for _ in range(expected_length)]
    elif isinstance(lr, (list, tuple)):
        if len(lr) != expected_length:
            raise ValueError(
                "Expected `{}` to have same length of optimizer's \
                    param groups {}, but got {}".format(
                    name, expected_length, len(lr)
                )
            )
    lr = [float(i) for i in lr]
    return lr


class LRScheduler:
    __IGNORE_KEYS__ = ["optimizer"]

    def __init__(  # pylint: disable=too-many-branches
        self, optimizer: Optimizer, current_epoch: int = -1
    ):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(
                "optimizer argument given to the lr_scheduler"
                "should be Optimizer, but got {}".format(type(optimizer))
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

        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        self._last_lr = None
        self.ignore_keys = LRScheduler.__IGNORE_KEYS__ + self.__IGNORE_KEYS__
        self.step()

    def _save_check(self, state_dict):
        return state_dict

    def _load_check(self, state_dict):
        return state_dict

    def state_dict(self):
        r"""Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in self.ignore_keys
        }
        self._save_check(state_dict)
        return state_dict

    def load_state_dict(self, state_dict):
        r"""Loads the schedulers state.

        Args:
            state_dict: scheduler state.
        """
        self._load_check(state_dict)
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

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class CosineAnnealingLR(LRScheduler):
    """
    CosineAnnealingLR (https://arxiv.org/abs/1608.03983)
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        current_epoch (int, optional): the index of current epoch. Default: -1.
    """

    def __init__(self, optimizer, T_max, eta_min, current_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, current_epoch)

    def get_lr(self):
        if self.current_epoch == 0:
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
        current_epoch (int, optional): the index of current epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Union[Callable, Sequence[Callable]],
        current_epoch: int = -1,
    ):
        if not isinstance(lr_lambda, Sequence):
            lr_lambda = [lr_lambda] * len(optimizer.param_groups)
        elif len(lr_lambda) != len(optimizer.param_groups):
            raise ValueError(
                "Expected `lr_lambda` to have the same length of \
                `param_groups`, but got {}".format(
                    len(lr_lambda)
                )
            )
        self.lr_lambda = lr_lambda

        self._func_names = [func.__name__ for func in lr_lambda]
        super().__init__(optimizer, current_epoch)

    def _load_check(self, state_dict):
        func_names = state_dict["_func_names"]

        def _check(names):
            if len(names) != len(self._func_names):
                return False
            for n in names:
                if n not in self._func_names:
                    return False
            return True

        if not _check(func_names):
            raise RuntimeError(
                "Unable to load the state_dict, expected function names {}"
                "but got {}".format(self._func_names, func_names)
            )

    def get_lr(self):
        return [
            base_lr * lamb(self.current_epoch)
            for base_lr, lamb in zip(self.base_lrs, self.lr_lambda)
        ]


class OneCycleLR(LRScheduler):
    """
    Sets the learning rate according to the one cycle learning rate scheduler.

    The scheduler adjusts the learning rate from an initial learning rate
    to the maximum learning rate and then from that maximum learning rate
    to the minimum learning rate, which is much less than the initial learning rate.

    It has been proposed in `Super-Convergence: Very Fast Training of
    Neural Networks Using Large Learning Rates <https://arxiv.org/abs/1708.07120>`_.
    Please note that the default behaviour of this scheduler follows the
    fastai implementation of one cycle, which claims that
    “unpublished work has shown even better results by using only two phases”.

    If you want the behaviour of this scheduler to be consistent with the paper,
        please set ``three_phase=True`` .

    Also note that you should update learning rate each step.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float | List[float]): The maximum learning rate. It is a python float number.
            Functionally, it defines the initial learning rate by ``divide_factor`` .
        total_steps (int): Number of total training steps.
        divide_factor (float, optional): Initial learning rate will be determined by
            initial_learning_rate = max_learning_rate / divide_factor. Default: 25.
        end_learning_rate (float, optional): The minimum learning rate during training,
            it should be much less than initial learning rate.
        phase_pct (float): The percentage of total steps which used to increasing
            learning rate. Default: 0.3.
        anneal_strategy (str, optional): Strategy of adjusting learning rate.'cos'
            for cosine annealing,  'linear' for linear annealing. Default: 'cos'.
        three_phase (bool, optional): Whether to use three phase.
            If ``True``:
                1. The learning rate will first increase from initial learning rate
                    to maximum learning rate.
                2. Then it will decrease to initial learning rate. Number of step
                    in this phase is the same as the one in first phase.
                3. Finally, it will decrease to minimum learning rate which is much
                    less than initial learning rate.
            If ``False``:
                1. The learning rate will increase to maximum learning rate.
                2. Then it will directly decrease to minimum learning rate.
        last_epoch (int, optional):  The index of last epoch. Can be set to
            restart training. Default: -1, means initial learning rate.
        current_epoch (int, optional): the index of current epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: Union[Number, Sequence[Number]],
        total_steps: int,
        divide_factor: float = 25.0,
        end_lr: Union[Number, Sequence[Number]] = 0.0001,
        phase_pct: float = 0.3,
        anneal_strategy: str = "cos",
        three_phase: bool = False,
        current_epoch=-1,
    ):
        self._lr_length = len(optimizer.param_groups)
        _check_valid(
            [total_steps, divide_factor],
            lambda x: x > 0,
            "Expected value to be greater than 0",
        )
        _check_valid(
            phase_pct, lambda x: 1 > x > 0, "Expected value to be between 0 and 1"
        )

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

        max_lr = _check_lrs(max_lr, "max_lr", self._lr_length)
        end_lr = _check_lrs(end_lr, "end_lr", self._lr_length)

        self.total_steps = total_steps
        self.max_lr = max_lr

        initial_lr = [m / divide_factor for m in max_lr]

        if current_epoch == -1:
            for idx, group in enumerate(optimizer.param_groups):
                group["max_lr"] = max_lr[idx]
                group["end_lr"] = end_lr[idx]
                group["initial_lr"] = initial_lr[idx]

        if three_phase:
            # Note: not sure for now
            if phase_pct >= 0.5:
                raise ValueError(
                    "When `three_phase` is True, 'phase_pct' must be less than 0.5"
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
            self._lr_config = [initial_lr, max_lr, initial_lr, end_lr]
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
            self._lr_config = [initial_lr, max_lr, end_lr]

        self._config_length = len(self._lr_config)
        self._lr_config = [
            [config[i] for config in self._lr_config] for i in range(self._lr_length)
        ]

        super().__init__(optimizer, current_epoch)

    @staticmethod
    def _cos_annealing(start_lr, end_lr, pct):
        cos_out = math.cos(math.pi * pct) + 1
        return end_lr + (start_lr - end_lr) / 2.0 * cos_out

    @staticmethod
    def _linear_annealing(start_lr, end_lr, pct):
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
            # i == len(self._lr_config[0]) - 2 catch the last step
            # otherwise it will return None.
            if current_step <= end_step or i == self._config_length - 2:
                # self._step_config[i] means start step of a phase.
                percentage = (current_step - self._step_config[i]) / step_size
                return [
                    self.anneal_func(config[i], config[i + 1], percentage)
                    for config in self._lr_config
                ]


class CyclicLR(LRScheduler):
    r"""
    Set the learning rate according to the cyclic learning rate (CLR) scheduler.
    The scheduler regards the process of learning rate adjustment
    as one cycle after another. It cycles the learning rate between
    two boundaries with a constant frequency. The distance between
    the two boundaries can be scaled on  a per-iteration or per-cycle basis.

    It has been proposed in `Cyclic Learning Rates for
        Training Neural Networks <https://arxiv.org/abs/1506.01186>`_.

    According to the paper, the cyclic learning rate schedule has
        three build-in scale methods:

    * "triangular": A basic triangular cycle without any amplitude scaling.
    * "triangular2": A basic triangular cycle that reduce initial amplitude
        by half each cycle.
    * "exp_range": A cycle that scales initial amplitude by scale function
        which is defined as :math:`gamma^{iterations}` .

    The initial amplitude is defined as max_learning_rate - base_learning_rate.
    Also note that you should update learning rate each step.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float | List[float]): Maximum learning rate in the cycle.
            It defines the cycle amplitude as above. Since there is
            some scaling operation during process of learning rate adjustment,
            max_lr may not actually be reached.
        step_size_up (int): Number of training steps, which is
            used to increase learning rate in a cycle.
            The step size of one cycle will be defined
            by step_size_up + step_size_down. According to the paper,
            step size should be set as at least 3 or 4 times steps in one epoch.
        step_size_down (int, optional): Number of training steps,
            which is used to decrease learning rate in a cycle.
            If not specified, it's value will initialize to `` step_size_up `` .
            Default: None
        mode (str, optional): one of 'triangular', 'triangular2' or 'exp_range'.
            If scale_fn is specified, this argument will be ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function: gamma**iterations.
            Used only when mode = 'exp_range'. Default: 1.0
        base_lr (float | List[float], optional): Initial learning rate,
            which is the lower boundary in the cycle. The paper recommends
            that set the base_lr to 1/3 or 1/4 of max_learning_rate.
        scale_fn (function, optional): A custom scaling function, which is
            used to replace three build-in methods. It should only have one argument.
            For all x >= 0, 0 <= scale_fn(x) <= 1. If specified,
            then 'mode' will be ignored. Default: None
        scale_mode (str, optional): One of 'cycle' or 'iterations'. Defines whether
            scale_fn is evaluated on cycle number or cycle iterations
            (total iterations since start of training). Default: 'cycle'
        current_epoch (int, optional): the index of current epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: Union[Number, Sequence[Number]],
        step_size_up: int,
        step_size_down: Optional[int] = None,
        mode: str = "triangular",
        gamma: Number = 1.0,
        base_lr: Optional[Union[Number, Sequence[Number]]] = None,
        scale_fn: Optional[Callable] = None,
        scale_mode: str = "cycle",
        current_epoch: int = -1,
    ):
        self._lr_length = len(optimizer.param_groups)
        _check_valid(
            [step_size_up], lambda x: x > 0, "Expected value to be greater than 0"
        )

        # do not use base_lrs
        if base_lr is None:
            self.base_lr = [group["lr"] for group in optimizer.param_groups]
        else:
            self.base_lr = _check_lrs(base_lr, "base_lr", self._lr_length)
            for lr, group in zip(self.base_lr, optimizer.param_groups):
                group["lr"] = lr

        self.max_lr = _check_lrs(max_lr, "max_lr", self._lr_length)

        self.step_size_up = step_size_up
        self.step_size_down = step_size_up if step_size_down is None else step_size_down

        self.cycle_size = step_size_up + self.step_size_down
        self.step_up_pct = step_size_up / self.cycle_size
        self.amplitude = [
            self.max_lr[i] - self.base_lr[i] for i in range(self._lr_length)
        ]

        if mode not in ["triangular", "triangular2", "exp_range"] and scale_fn is None:
            raise ValueError(
                """'mode' is invalid and 'scale_fn' is not specified,
                make sure one of 'mode' or 'scale_fn' is valid"""
            )
        if scale_mode not in ["cycle", "iterations"]:
            raise ValueError("'scale_mode' must be one of 'cycle' or 'iterations")

        self.mode = mode
        self.gamma = gamma  # only for exp_range mode

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

    @staticmethod
    def _triangular_scale_fn(_):
        return 1.0

    @staticmethod
    def _triangular2_scale_fn(x):
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
            # scale_factor = (1 - pct_per_cycle) / (1 - self.step_up_pct)
            # for higher precision
            scale_factor = (pct_per_cycle - 1) / (self.step_up_pct - 1)

        base_height = [scale_factor * amp for amp in self.amplitude]

        if self.scale_mode == "cycle":
            scale = self.scale_fn(cycle)
        else:
            scale = self.scale_fn(iterations)

        return [
            scale * base_height[i] + self.base_lrs[i] for i in range(self._lr_length)
        ]
