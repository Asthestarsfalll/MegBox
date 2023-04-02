from copy import deepcopy
from typing import Callable, Dict, List, Sequence, Union

import megengine as mge
import megengine.module as M
from megengine.module import init
from tqdm import tqdm

from megbox.types import ModuleType


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


def _test_modules(
    module_mappers: Union[Dict[str, ModuleType], List[Dict[str, ModuleType]]],
    kwargs_mappers: Union[Dict[str, List], List[Dict[str, List]]],
    spatial_sizes: Sequence[int],
    check_func: Callable,
):
    if not isinstance(module_mappers, list):
        module_mappers = [module_mappers]
    if not isinstance(kwargs_mappers, list):
        kwargs_mappers = [kwargs_mappers] * len(module_mappers)
    names = tqdm(module_mappers[0].keys())

    def run(is_gpu: bool):
        for name in names:
            names.set_description("test module {}".format(name))
            module_classes = [mapper[name] for mapper in module_mappers]
            _default = deepcopy(kwargs_mappers[0][name])
            kwargs_lists = [mapper.get(name, _default) for mapper in kwargs_mappers]
            if len(module_classes) == 1:
                module_classes = module_classes[0]
            for i in range(len(kwargs_lists[0])):
                kwargs = [k[i] for k in kwargs_lists]
                if len(kwargs) == 1:
                    kwargs = kwargs[0]
                for sp_size in spatial_sizes:
                    check_func(
                        module_classes,
                        name=name,
                        kwargs=kwargs,
                        sp_size=sp_size,
                        is_gpu=is_gpu,
                    )
                del kwargs

    if mge.is_cuda_available():
        run(True)
    mge.set_default_device("cpu0")
    run(False)
