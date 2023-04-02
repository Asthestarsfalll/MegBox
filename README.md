![logo](logo.png)

## Introduction

`MegBox` is an easy-to-use, well-rounded and safe toolbox of MegEngine. Aim to imporving usage experience and speeding up develop process.


`MegBox` is still in an early development stage.

## Features

### easy_use

<details>
  <summary>Easily generate tensor</summary>

  ```python
  from megbox import easy_use

  x = easy_use.randn(2, 3, 4, 5, 6)
  ```

</details>


<details>
  <summary>Easily pad</summary>

  ```python
  y1 = F.nn.pad(x, [(0, 0), (0, 0), (0, 0), (0, 1), (1, 0)])
  y2 = easy_use.pad(x, [1, 0, 0, 1])

  print(easy_use.all(y1 == y2))
  ```

</details>


<details>
  <summary>Easily exchang axes</summary>

  ```python
  y1 = x.transpose(0, 1, 2, 4, 3)
  y2 = easy_use.exchang_axes(x, -2, -1)

  print(easy_use.all(y1 == y2))
  ```

</details>


<details>
  <summary>Easily use where</summary>

  ```python
  # use number in where
  y1 = F.where(x > 0, x, mge.tensor(0))
  y2 = easy_use.where(x > 0, x, 0)

  print(easy_use.all(y1 == y2))
  ```

</details>


### well-rounded

<details>
  <summary>Support Pooling with ceil mode</summary>

  ```python
  from megbox.module import AvgPool2d, MaxPool2d

  module = MaxPool2d(
      kernel_size=2,
      ceil_mode=True,
  )

  # Note: Use an approximate implementation, which may cause some problem.
  module = AvgPool2d(
      kernel_size=2,
      ceil_mode=True,
  )
  ```

</details>


<details>
  <summary>Be aligned with torch's implementation</summary>

  ```python
  from megbox.module import AdaptiveAvgPool2d, AdaptiveMaxPool2d

  module = AdaptiveAvgPool2d(7)

  module = AdaptiveMaxPool2d(3)
  ```

</details>


<details>
  <summary>Commonly used attention block</summary>

  ```python
  from megbox import attention

  print(attention.__all__)
  se = attention.SEBlock(in_channels=64, reduction=16)
  ```

</details>


<details>
  <summary>Some kinds of convolution variants</summary>

  ```python
  from megbox import conv

  print(conv.__al__)
  involution = conv.Involution(channels=64, kernel_size=11, stride=1)
  ```

</details>


<details>
  <summary>Further support for reparameterization convolution with dilation</summary>

  ```python
  from megbox.reparam import RepConv2d, RepLargeKernelConv2d

  rep_conv = RepConv2d(32, 32, dilation=(1, 2))
  rep_lk_conv = RepLargeKernelConv2d(
      channels=32,
      kernel_size=11,
      small_kernel_size=(5, 1),
      dilation=2,
  )

  rep_conv.switch_to_deploy()
  rep_lk_conv.switch_to_deploy()
  ```

</details>

<details>
  <summary>Visualize the reparameterization process</summary>

  ```python
  from megbox.reparam import visualize

  visualize(kernel_sizes=(7, 5, 3), dilations=(2, 3, 1), save_dir='./')
  ```

</details>


### safe


<details>
  <summary>Safely sort with NaN</summary>

  ```python
  import megengine.functional as F
  from megbox.functional.safe import sort
  import megengine as mge

  x = mge.tensor([3., 4., 2., float("NaN"), 1., 2., float("NaN")])

  # can not return corrct result
  y1 = F.sort(x)
  y2 = sort(x)
  ```

</details>


## Details

**More details can be found in documents(will be supported as soon as possible).**

TODO:

## Reference

[timm](https://github.com/rwightman/pytorch-image-models)

[External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch#21-Polarized-Self-Attention-Usage)
