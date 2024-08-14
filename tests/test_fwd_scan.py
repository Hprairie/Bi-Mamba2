# Copyright (c) 2024, Hayden Prairie.

import torch
import pytest

from fixtures.fwd import fwd_compare

@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("seqlen", [32])
@pytest.mark.parametrize("nheads", [1])
@pytest.mark.parametrize("ngroups", [1])
@pytest.mark.parametrize("chunk_size", [32])
@pytest.mark.parametrize("headdim", [1])
@pytest.mark.parametrize("dstate", [1])
@pytest.mark.parametrize("delta_softplus", [True])
@pytest.mark.parametrize("dtype", [torch.float32])
class TestFwd:
    def test_fwd(self, fwd_compare):
        assert fwd_compare, "Fwd Test Failed"
