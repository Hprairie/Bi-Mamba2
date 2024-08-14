# A Bi-Directional Extension of Mamba2

Several works such as Hydra and MambaMixer have formulated bidirectionality through qusiseperable matrices. However, they often flip and rerun Mamba2's GPU kernel twice. This is too slow, and we can get a lot better performance if we fuse the kernel together. I edited the Triton Kernel from Mamba2 so that they are bidirectional. This will save both memory space and also time.

## NOTE

I still haven't finished this kernel or benchmarked it. I will remove this notice when I do so.

- Fwd is almost done I will check it off the TODO when all tests are passing
- I am still writing the bwd, and will post updates as I go

# Project Structure and Install

To access the kernels, run:

```shell
pip install -e .
```
You can access the normal `ssd` kernels through `ssd.uni`. You can access the bidirectional kernels through `ssd.bi`.

## TODO:

- [x] Write FWD Implementation
- [ ] Debug and Test FWD Implemntation
- [ ] Write BWD Implementation
- [ ] Debug and Test BWD Implementation

## Benchmarking

Comparing the Bi-Mamba2 optimized kernel, to the Naive approach of flipping and accumulating the sequence, we get the following.

![Bi-Mamba2](assets/Naive_Comparison.png)

