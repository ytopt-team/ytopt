Autotuning Apache TVM (Tensor Virtual Machine)-based scientific applications (3mm, lu, cholesky)
using ytopt and AutoTVM. See the details about [TVM](https://tvm.apache.org) and [AutoTVM](https://tvm.apache.org/docs/how_to/tune_with_autotvm/index.html).

# Directory

This directory includes 3mm, lu, and cholesky. See the [AI4S'23 paper](https://arxiv.org/pdf/2309.07235.pdf) for the details.

```
3mm/
    Autotuning 3mm with two problem sizes: L and XL (3mm_TVM/ using AutoTVM; 3mm_ytopt/ using ytopt)
lu/
    Autotuning lu with two problem sizes: L and XL (LU_TVM/ using AutoTVM; LU_ytopt/ using ytopt)
cholesky/
    Autotuning cholesky with two problem sizes: L and XL (cholesky_TVM/ using AutoTVM; cholesky_ytopt/ using ytopt)
```

