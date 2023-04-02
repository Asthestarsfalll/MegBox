from typing import Union

from megengine.module.pooling import Module, _PoolNd

__all__ = [
    "Number",
    "PoolType",
]

Number = Union[int, float]
PoolType = type(_PoolNd)
ModuleType = type(Module)
