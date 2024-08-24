# Copyright (c) 2024, Hayden Prairie.

import torch
import pytest

from fixtures.chunk_scan_bwd_dstates import chunk_scan_bwd_dstate_compare 
from fixtures.chunk_scan_chunk_state_bwd_dx import chunk_scan_bwd_dx_compare
from fixtures.chunk_scan_bwd_dc import chunk_scan_bwd_dc_compare 
from fixtures.chunk_scan_bwd_dcb import chunk_state_bwd_dcb_compare
from fixtures.chunk_scan_bwd_ddAcs_stable_bwd import chunk_scan_bwd_ddAcs_stable_compare


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


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("seqlen", [64])
@pytest.mark.parametrize("nheads", [1])
@pytest.mark.parametrize("chunk_size", [64])
@pytest.mark.parametrize("ngroups", [1])
@pytest.mark.parametrize("headdim", [1, 2, 32])
@pytest.mark.parametrize("dstate", [1, 2, 32])
@pytest.mark.parametrize("dtype", [torch.float32])
class TestChunkScanBwdDx:
    def test_chunk_scan(self, chunk_scan_bwd_dx_compare):
        assert chunk_scan_bwd_dx_compare, "Chunk Scan Dx Bwd Test Failed"

@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("seqlen", [64])
@pytest.mark.parametrize("nheads", [1])
@pytest.mark.parametrize("chunk_size", [64])
@pytest.mark.parametrize("ngroups", [1])
@pytest.mark.parametrize("headdim", [1, 2, 32])
@pytest.mark.parametrize("dstate", [1, 2, 32])
@pytest.mark.parametrize("dtype", [torch.float32])
class TestChunkScanBwdDc:
    def test_chunk_scan(self, chunk_scan_bwd_dc_compare):
        assert chunk_scan_bwd_dc_compare, "Chunk Scan Dc Bwd Test Failed"


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("seqlen", [128])
@pytest.mark.parametrize("nheads", [1])
@pytest.mark.parametrize("chunk_size", [128])
@pytest.mark.parametrize("ngroups", [1])
@pytest.mark.parametrize("headdim", [1, 2, 32])
@pytest.mark.parametrize("dstate", [1, 2, 32])
@pytest.mark.parametrize("dtype", [torch.float32])
class TestChunkScanBwdDcb:
    def test_chunk_scan(self, chunk_state_bwd_dcb_compare):
        assert chunk_state_bwd_dcb_compare, "Chunk Scan Dcb Bwd Test Failed"

@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("seqlen", [128])
@pytest.mark.parametrize("nheads", [1])
@pytest.mark.parametrize("chunk_size", [128])
@pytest.mark.parametrize("ngroups", [1])
@pytest.mark.parametrize("headdim", [1, 2, 32])
@pytest.mark.parametrize("dstate", [1, 2, 32])
@pytest.mark.parametrize("dtype", [torch.float32])
class TestChunkScanBwdDdAcsStable:
    def test_chunk_scan(self, chunk_scan_bwd_ddAcs_stable_compare):
        assert chunk_scan_bwd_ddAcs_stable_compare, "Chunk Scan DdAcs Bwd Stable Bwd Test Failed"
