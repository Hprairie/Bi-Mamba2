# Copyright (c) 2024, Hayden Prairie.

import torch
import pytest

from fixtures.chunk_scan_bwd_dstates import chunk_scan_bwd_dstate_compare 


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("seqlen", [8, 64, 128, 256])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@pytest.mark.parametrize("ngroups", [4])
@pytest.mark.parametrize("headdim", [1, 2, 32])
@pytest.mark.parametrize("dstate", [1, 2, 32])
@pytest.mark.parametrize("dtype", [torch.float32])
class TestChunkScanBwdDstate:
    def test_chunk_scan(self, chunk_scan_bwd_dstate_compare):
        assert chunk_scan_bwd_dstate_compare, "Chunk Scan Dstate Bwd Test Failed"
