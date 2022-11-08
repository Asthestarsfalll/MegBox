from megengine import Tensor
from megengine.module import Dropout

from ..functional.elementwise import drop_path


class DropPath(Dropout):
    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)
