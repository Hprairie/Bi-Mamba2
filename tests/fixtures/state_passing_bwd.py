# Copyright (c) 2024, Hayden Prairie.

import pytest

import torch
from torch.testing import assert_close

from ssd.uni.ssd_state_passing import _state_passing_bwd
from ssd.bi.ssd_state_passing import _state_passing_bwd as _state_passing_bwd_bi

@pytest.fixture
def create_bwd_state_passing_tensors(batch, nchunks, nheads, dim, dtype):
    torch.random.manual_seed(0)
    device = torch.device("cuda")
    states_f = (0.5 * torch.randn(batch, nchunks, nheads, dim)).to(device=device, dtype=torch.float32).requires_grad_()
    states_b = (0.5 * torch.randn(batch, nchunks, nheads, dim)).to(device=device, dtype=torch.float32).requires_grad_()
    dstates_f = torch.rand_like(states_f)
    dstates_b = torch.rand_like(states_b)
    dA_cumsum_f = (0.5 * torch.randn(batch, nheads, nchunks)).to(device=device, dtype=torch.float32).requires_grad_()
    dA_cumsum_b = (0.5 * torch.randn(batch, nheads, nchunks)).to(device=device, dtype=torch.float32).requires_grad_()

    return (
            states_f,
            states_b,
            dstates_f,
            dstates_b,
            dA_cumsum_f,
            dA_cumsum_b,
    )

def create_ref_tensors(
        states_f,
        states_b,
        dstates_f,
        dstates_b,
        dA_cumsum_f,
        dA_cumsum_b,
):
    states_f_ref = states_f.detach().clone().requires_grad_()
    states_b_ref = states_b.detach().clone().requires_grad_()
    dstates_f_ref = dstates_f.detach().clone().requires_grad_()
    dstates_b_ref = dstates_b.detach().clone().requires_grad_()
    dA_cumsum_f_ref = dA_cumsum_f.detach().clone().requires_grad_()
    dA_cumsum_b_ref = dA_cumsum_b.detach().clone().requires_grad_()

    return states_f_ref, states_b_ref, dstates_f_ref, dstates_b_ref, dA_cumsum_f_ref, dA_cumsum_b_ref


@pytest.fixture
def state_bwd_passing_compare(create_bwd_state_passing_tensors):
    states_f, states_b, dstates_f, dstates_b, dA_cumsum_f, dA_cumsum_b = create_bwd_state_passing_tensors 
    states_f_ref, states_b_ref, dstates_f_ref, dstates_b_ref, dA_cumsum_f_ref, dA_cumsum_b_ref = create_ref_tensors(*create_bwd_state_passing_tensors)

    dstates_out_f, ddA_chunk_cumsum_f, _ = _state_passing_bwd(states_f, dA_cumsum_f, dstates_f)
    dstates_out_b, ddA_chunk_cumsum_b, _ = _state_passing_bwd(states_b.flip([1]), dA_cumsum_b.flip([2]), dstates_b.flip([1]))
    dstates_out_b = dstates_out_b.flip([1])
    ddA_chunk_cumsum_b = ddA_chunk_cumsum_b.flip([2])

    dstates_out_f_ref, ddA_chunk_cumsum_f_ref = _state_passing_bwd_bi(states_f_ref, dA_cumsum_f_ref, dstates_f_ref)
    dstates_out_b_ref, ddA_chunk_cumsum_b_ref = _state_passing_bwd_bi(states_b_ref, dA_cumsum_b_ref, dstates_b_ref, reverse=True)

    rtol, atol = (6e-4, 2e-3)

    assert_close(dstates_out_f, dstates_out_f_ref, rtol=rtol, atol=atol)
    assert_close(ddA_chunk_cumsum_f, ddA_chunk_cumsum_f_ref, rtol=rtol, atol=atol)
    assert_close(dstates_out_b, dstates_out_b_ref, rtol=rtol, atol=atol)
    assert_close(ddA_chunk_cumsum_b, ddA_chunk_cumsum_b_ref, rtol=rtol, atol=atol)

    return True
