from megengine import Parameter, Tensor
from megengine import functional as F
from megengine.module import Module


class LayerScale(Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gamma = Parameter(init_values * F.ones((dim)))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma
