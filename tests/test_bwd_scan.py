# Copyright (c) 2024, Hayden Prairie.

import torch
import pytest

from fixtures.bwd import bwd_compare

@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("seqlen", [32, 64, 128])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("ngroups", [4])
@pytest.mark.parametrize("chunk_size", [32])
@pytest.mark.parametrize("headdim", [4, 16])
@pytest.mark.parametrize("dstate", [4, 16])
@pytest.mark.parametrize("delta_softplus", [True])
@pytest.mark.parametrize("dtype", [torch.float32])
class TestBwd:
    def test_bwd(self, bwd_compare):
        assert bwd_compare, "Fwd Test Failed"
