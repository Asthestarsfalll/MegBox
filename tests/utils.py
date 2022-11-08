from typing import Callable, Dict, List, Sequence

import megengine as mge
from megengine.module import Module


def _test_modules(
    module_mapper: Dict[str, Module],
    kwargs_mapper: Dict[str, List],
    spatial_sizes: Sequence[int],
    check_func: Callable,
):
    def run():
        for name, module_class in module_mapper.items():
            kwargs_list = kwargs_mapper[name]
            for kwargs in kwargs_list:
                module = module_class(**kwargs)
                module.eval()
                for sp_size in spatial_sizes:
                    check_func(module, name=name, kwargs=kwargs, sp_size=sp_size)

    if mge.is_cuda_available():
        run()
    mge.set_default_device("cpu0")
    run()
