import megengine.functional as F
import megengine.module as M
from megengine import Tensor


class Flatten(M.Module):
    def __init__(self, start_axis: int = 0, end_axis: int = -1) -> None:
        super(Flatten, self).__init__()
        self.start_axis = start_axis
        self.end_axis = end_axis

    def forward(self, x: Tensor) -> Tensor:
        return F.flatten(x, self.start_axis, self.end_axis)
