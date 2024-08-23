
import torch
import pytest

from fixtures.chunk_state_bwd_db import chunk_state_bwd_db_compare

@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("seqlen", [64])
@pytest.mark.parametrize("nheads", [1])
@pytest.mark.parametrize("chunk_size", [64])
@pytest.mark.parametrize("ngroups", [1])
@pytest.mark.parametrize("headdim", [1, 2, 32])
@pytest.mark.parametrize("dstate", [1, 2, 32])
@pytest.mark.parametrize("dtype", [torch.float32])
class TestChunkStateBwdDd:
    def test_chunk_state(self, chunk_state_bwd_db_compare):
        assert chunk_state_bwd_db_compare, "Chunk State Dd Bwd Test Failed"
