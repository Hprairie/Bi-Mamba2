# Copyright (c) 2024, Hayden Prairie.

import torch
import pytest

from fixtures.chunk_state_fwd import chunk_state_compare


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("seqlen", [8, 64, 128, 256])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@pytest.mark.parametrize("ngroups", [4])
@pytest.mark.parametrize("headdim", [1, 2, 32])
@pytest.mark.parametrize("dstate", [1, 2, 32])
@pytest.mark.parametrize("dtype", [torch.float32])
class TestChunkStateFwd:
    def test_chunk_state(self, chunk_state_compare):
        assert chunk_state_compare, "Chunk State Fwd Test Failed"
