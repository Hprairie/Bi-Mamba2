# Copyright (c) 2024, Hayden Prairie.

import pytest

import torch
from torch.testing import assert_close

from ssd.uni.ssd_state_passing import _state_passing_fwd
from ssd.bi.ssd_state_passing import _state_passing_fwd as _state_passing_fwd_bi

@pytest.fixture
def create_state_passing_tensors(batch, nchunks, nheads, dim, dtype):
    torch.random.manual_seed(0)
    device = torch.device("cuda")
    states_f = (0.5 * torch.randn(batch, nchunks, nheads, dim)).to(device=device, dtype=torch.float32).requires_grad_()
    states_b = (0.5 * torch.randn(batch, nchunks, nheads, dim)).to(device=device, dtype=torch.float32).requires_grad_()
    dA_cumsum_f = (0.5 * torch.randn(batch, nheads, nchunks)).to(device=device, dtype=torch.float32).requires_grad_()
    dA_cumsum_b = (0.5 * torch.randn(batch, nheads, nchunks)).to(device=device, dtype=torch.float32).requires_grad_()

    return (
            states_f,
            states_b,
            dA_cumsum_f,
            dA_cumsum_b,
    )

def create_ref_tensors(
        states_f,
        states_b,
        dA_cumsum_f,
        dA_cumsum_b,
):
    states_f_ref = states_f.detach().clone().requires_grad_()
    states_b_ref = states_b.detach().clone().requires_grad_()
    dA_cumsum_f_ref = dA_cumsum_f.detach().clone().requires_grad_()
    dA_cumsum_b_ref = dA_cumsum_b.detach().clone().requires_grad_()

    return states_f_ref, states_b_ref, dA_cumsum_f_ref, dA_cumsum_b_ref


@pytest.fixture
def state_passing_compare(create_state_passing_tensors):
    states_f, states_b, dA_cumsum_f, dA_cumsum_b = create_state_passing_tensors 
    states_f_ref, states_b_ref, dA_cumsum_f_ref, dA_cumsum_b_ref = create_ref_tensors(*create_state_passing_tensors)

    out_f, final_f = _state_passing_fwd(states_f, dA_cumsum_f)
    out_b, final_b = _state_passing_fwd(states_b.flip([1]), dA_cumsum_b.flip([2]))
    out_b = out_b.flip([1])

    out_f_ref, final_f_ref = _state_passing_fwd_bi(states_f_ref, dA_cumsum_f_ref)
    out_b_ref, final_b_ref = _state_passing_fwd_bi(states_f_ref, dA_cumsum_f_ref, reverse=True)

    rtol, atol = (6e-4, 2e-3)

    assert_close(states_f, states_f_ref, rtol=rtol, atol=atol)
    assert_close(states_b, states_b_ref, rtol=rtol, atol=atol)

    return True
