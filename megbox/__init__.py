from . import attention, block, data, functional, optimizer
from .block import conv
from .optimizer import lr_scheduler

__version__ = "0.1.1"
__author__ = "Asthestarsfalll"
__year__ = "2023"
__project_info__ = {
    "name": __name__,
    "version": __version__,
    "copyright": f"{__year__}, {__author__}",
    "author": __author__,
}

__all__ = [
    "attention",
    "block",
    "conv",
    "data",
    "functional",
    "lr_scheduler",
    "optimizer",
    "__version__",
    "__project_info__",
]
