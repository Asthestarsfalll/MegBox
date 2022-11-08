from typing import Tuple

from megengine import Tensor
from megengine.data.transform import Transform


class ToTensor(Transform):
    def apply(self, input: Tuple) -> Tuple[Tensor, ...]:
        return tuple((Tensor(input[0]), Tensor(input[1])))
