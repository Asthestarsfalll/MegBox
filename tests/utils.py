from typing import Callable, Dict, List, Sequence

import megengine as mge
import megengine.module as M
from megengine.module import Module, init


def _init_weights(m):
    if isinstance(m, (M.Conv2d, M.Linear)):
        init.ones_(m.weight)
        if m.bias is not None:
            init.ones_(m.bias)
    elif isinstance(m, M.BatchNorm2d) and m.affine:
        init.ones_(m.weight)
        init.ones_(m.bias)
        m.running_mean = mge.functional.full_like(m.running_mean, 0.5)
        m.running_var = mge.functional.full_like(m.running_var, 0.1)


MODULE_TYPE = type(Module)


def _test_modules(
    module_mapper: Dict[str, MODULE_TYPE],
    kwargs_mapper: Dict[str, List],
    spatial_sizes: Sequence[int],
    check_func: Callable,
):
    def run():
        for name, module_class in module_mapper.items():
            kwargs_list = kwargs_mapper[name]
            for kwargs in kwargs_list:
                for sp_size in spatial_sizes:
                    check_func(module_class, name=name, kwargs=kwargs, sp_size=sp_size)

    if mge.is_cuda_available():
        run()
    mge.set_default_device("cpu0")
    run()
