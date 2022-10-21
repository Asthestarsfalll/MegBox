# MegBox

![logo](logo.png)

## Introduction

`MegBox` is a easy-use, well-rounded and safe toolbox of `MegEngine`. Aim to imporving usage experience and speeding up develop process.



`MegBox` is still in early development stage.

## Features

### easy_use

Easily generate tensor:

```python
from megbox import esay_use

x = esay_use.randn(2, 3, 4, 5, 6)
```

Easily pad:

```python
y1 = F.nn.pad(x, [(0, 0), (0, 0), (0, 0), (0, 1), (1, 0)])
y2 = esay_use.pad(x, [1, 0, 0, 1])

print(esay_use.all(y1 == y2))
```

Easily exchang axes:

```python
y1 = x.transpose(0, 1, 2, 4, 3)
y2 = esay_use.exchang_axes(x, -2, -1)

print(esay_use.all(y1 == y2))
```

Easily use where:

```python
# use number in where
y1 = F.where(x > 0, x, mge.tensor(0))
y2 = esay_use.where(x > 0, x, 0)

print(esay_use.all(y1 == y2))
```

More details can be found in documents(will be spporter soon).

### well-rounded

Support Pooling with ceil mode:

```python
from megbox.module import AvgPool2d, MaxPool2d

module = AvgPool2d(
    kernel_size=2,
    ceil_mode=True,
)

module = MaxPool2d(
    kernel_size=2,
    ceil_mode=True,
)
```

Be aligned with torch's implementation:

```python
from megbox.module import AdaptiveAvgPool2D, AdaptiveMaxPool2D

module = AdaptiveAvgPool2D(7)

module = AdaptiveMaxPool2D(3)
```

Commonly used attention block and some kinds of convolution variants.

### safe

Safely sort with NaN:

```python
import megengine.functional as F
from megbox.functional.safe import sort
import megengine as mge

x = mge.tensor([3., 4., 2., float("NaN"), 1., 2., float("NaN")])

# can not return corrct result
y1 = F.sort(x)
y2 = sort(x)
```



## Details

TODO:

## TODO

- [ ]  code style and sort imports
- [ ] add pre-commit  
- [ ] add tests
- [ ] add documents
- [ ] complete README
- [ ] ...

## Reference

[timm](https://github.com/rwightman/pytorch-image-models)
[External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch#21-Polarized-Self-Attention-Usage)
