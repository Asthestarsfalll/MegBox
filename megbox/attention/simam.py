import megengine.module as M
from megengine import Tensor
from megengine.functional import pow


class SimAM(M.Module):
    def __init__(self, e_lambda: float = 1e-4) -> None:
        super(SimAM, self).__init__()

        self.activaton = M.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x: Tensor) -> Tensor:

        h, w = x.shape[-2:]

        n = w * h - 1

        x_minus_mu_square = pow(x - x.mean(axis=[2, 3], keepdims=True), 2)
        y = (
            x_minus_mu_square
            / (
                4
                * (
                    x_minus_mu_square.sum(axis=[2, 3], keepdims=True) / n
                    + self.e_lambda
                )
            )
            + 0.5
        )

        return x * self.activaton(y)
