# Copyright (c) 2024, Hayden Prairie.

import torch
import pytest

from fixtures.cumsum_fwd import cumsum_compare

@pytest.mark.parametrize("batch", [2])
@pytest.mark.parametrize("seqlen", [2, 8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("nheads", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@pytest.mark.parametrize("has_delta_bias", [True, False])
@pytest.mark.parametrize("softplus", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32])
class TestChunkCumsumFwd:
    def test_cumsum(self, cumsum_compare):
        assert cumsum_compare, "Chunk Cumsum Fwd Test Failed"


