# Copyright (c) 2024, Hayden Prairie.

import torch
import pytest

from fixtures.state_passing_fwd import state_passing_compare


@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("nchunks", [2, 4, 8])
@pytest.mark.parametrize("nheads", [4])
@pytest.mark.parametrize("dim", [1, 2, 32])
@pytest.mark.parametrize("dtype", [torch.float32])
class TestStatePassingFwd:
    def test_state_passing(self, state_passing_compare):
        assert state_passing_compare, "Chunk State Fwd Test Failed"
