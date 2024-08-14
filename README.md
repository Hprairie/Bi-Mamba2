# A Bi-Directional Extension of Mamba2

Several works such as Hydra and MambaMixer have formulated bidirectionality through qusiseperable matrices. However, they often flip and rerun Mamba2's GPU kernel twice. This is too slow, and we can get a lot better performance if we fuse the kernel together. I edited the Triton Kernel from Mamba2 so that they are bidirectional. This will save both memory space and also time.

## NOTE

I still haven't finished this kernel or benchmarked it. I will remove this notice when I do so.

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

Benchmarking the Forward Pass saves a lot of time.

![Bi-Mamba2](assets/Bidirectional_Comparison.png)
