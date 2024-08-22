<h1 align="center" style="fontsize:50em"><b>A Bi-Directional Extension of Mamba2</b></h1>

Several works such as Hydra and MambaMixer have formulated bidirectionality through qusiseperable matrices. I highly recommend reading both of these papers to understand how bi-directionality can be done with Mamba. Unfortunately both of these implementations don't have an optimized kernel, which often increases the training and inference time by more than **2x**.

To over come this issue, I wrote the following GPU kernel which both reduces the memory overhead and the latency. It does so by fusing kernels together so that we minimize the number of loads and stores from global memory.

# NOTE

The fwd kernel is implemented, tested, and benchmarked. The bwd kernel is still in progress. I will remove this note when it's done. The bwd is written, just needs to be debugged and tested.

# A Brief Overview of Bidirectionality in Mamba

The idea of bidirectionality is to formulate the "Attention Matrix" as a quasiseperable matrix, meaning that the matrix can be decomposed into two seperable matrices, and a diagonal matrix. The forumulation is still subquadratic, as both seperable matrices and the diagonal matrix can be computed linearly. Hydra formulates the quasiseperable matrix in the following format:

$$ y = shift(SS(x)) + flip(shift(SS(flip(x)))) + Dx $$

This kernel does formulates the quasiseperable matrix as the following:

$$ y = SS(x) + flip(SS(flip(x))) + Dx $$

**Why?**: The main reasoning is simplicity. The shift opperation adds a lot of complexity to the kernel, and furthermore, shifting in SRAM is not currently supported by Triton. As I don't want to rewrite the entire kernel in CUDA, I compromise with the above formulation.

# Project Structure and Install

### Installing Locally

To access the kernels, run:

```shell
pip install -e .
```

You can access the normal `ssd` kernels through `ssd.uni`. You can access the bidirectional kernels through `ssd.bi`.

### Installing with PyPi

Coming soon.

# TODO:

- [x] Write FWD Implementation
- [x] Debug and Test FWD Implemntation
- [x] Write BWD Implementation
- [ ] Debug and Test BWD Implementation
- [ ] Create PyPi Package
- [ ] Add more benchmarks

# Modules and API

There is two ways to access the Bi-directional kernel. The first is through the functional defenition. For example if you want to run a chunk-wise bi-directional selective scan, you can do so with the following snippet:

**Causal Kernel**

```python
from ssd import ssd_selective_scan

```

**Bi-Directional Kernel**

```python
from ssd import bi_ssd_selective_scan

```

Alternatively you can also access it through a Module API, which is similar to a Mamba2 Layer:

**Causal Kernel**

```python
from ssd import Mamba2

layer = Mamba2(
    causal=True
)
```

**Causal Kernel**

```python
from ssd import Mamba2

layer = Mamba2(
    causal=False
)
```

**Note** Currently using `seq_idx` like in Mamba2 causal is unsupported. Additionally `passing init_hidden_states` is also unsupported.

# Benchmarking

The benchmarking code can be found in the `benchmark` folder. It can be run by using the following commmand:

```shell
python benchmark/benchmark_fwd_naive.py
```

## AMD 7900 XTX and 3970X Threadripper

Comparisson of fwd pass of Bi-Mamba2 v. Naive Flipping and Running Mamba2 kernel twice.
<p align="center">
  <img src="assets/Naive_Comparison.png" width="800" />
</p>

Comparison of fwd pass of Bi-Mamba2 v. causal Mamba2.

<p align="center">
  <img src="assets/Causal_Comparison.png" width="800" />
</p>

Comparison of bwd pass of Bi-Mamba2 v. Naive Flipping and Running Mamba2 kernel twice.

...

Comparison of bwd pass of Bi-Mamba2 v. causal Mamba2.

...

## 4060 Ti and Ryzen 9 8945HS

Coming Soon

## A100 40GB and Epyc 7763

Coming Soon

## H200 and Grace Chip

Coming Soon

# Tests

To run a test, simply use pytest along with the specific test file. For example, to run a test for the fwd pass of the kernel, use:

```shell
python -m pytest -x -s -v tests/test_fwd_scan.py::TestFwd
```

# Citation

If you find this kernel useful please cite Mamba, Hydra, and MambaMixer (They are amazing works!).

Give this repo a star also :)
