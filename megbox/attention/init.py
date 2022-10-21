from megengine.module import BatchNorm1d, BatchNorm2d, Conv2d, Linear, Module
from megengine.module.init import msra_normal_, normal_, ones_, zeros_


def _init_weights(module) -> None:
    if isinstance(module, Conv2d):
        msra_normal_(module.weight, mode="fan_out")
        if module.bias is not None:
            zeros_(module.bias)
    elif isinstance(module, Linear):
        normal_(module.weight, std=0.001)
        if module.bias is not None:
            zeros_(module.bias)
    elif isinstance(module, (BatchNorm2d, BatchNorm1d)):
        ones_(module.weight)
        zeros_(module.bias)
