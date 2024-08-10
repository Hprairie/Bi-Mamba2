# A Bi-Directional Extension of Mamba2

Several works such as Hydra and MambaMixer have formulated bidirectionality through qusiseperable matrices. However, they often flip and rerun Mamba2's GPU kernel twice. This is too slow, and we can get a lot better performance if we fuse the kernel together. I edited the Triton Kernel from Mamba2 so that they are bidirectional. This will save both memory space and also time.
