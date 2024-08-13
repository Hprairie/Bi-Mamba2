# Copyright (c) 2024, Hayden Prairie.

import pytest

import torch
from torch.testing import assert_close

import triton

from ssd.uni.ssd_chunk_state import _chunk_state_fwd 
from ssd.bi.ssd_chunk_state import _chunk_state_fwd as _chunk_state_fwd_bi 

@pytest.fixture
def create_chunk_state_scan_tensors(batch, seqlen, nheads, chunk_size, ngroups, headdim, dstate, dtype):
    torch.random.manual_seed(0)
    device = torch.device("cuda")
    assert nheads % ngroups == 0
    nchunks = triton.cdiv(seqlen, chunk_size)
    delta = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    dA_cumsum_f = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    dA_cumsum_b = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    B = (-0.5 * torch.randn(batch, seqlen, ngroups, dstate).abs()).to(device=device, dtype=torch.float32).requires_grad_()
    x = (0.5 * torch.randn(batch, seqlen, nheads, headdim)).to(device=device, dtype=torch.float32).requires_grad_()

    return (
            B,
            x,
            delta,
            dA_cumsum_f,
            dA_cumsum_b,
    )

def create_ref_tensors(
    B,
    x,
    delta,
    dA_cumsum_f,
    dA_cumsum_b,
):
    B_ref = B.detach().clone().requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    delta_ref = delta.detach().clone().requires_grad_()
    dA_cumsum_f_ref = dA_cumsum_f.detach().clone().requires_grad_()
    dA_cumsum_b_ref = dA_cumsum_b.detach().clone().requires_grad_()

    return B_ref, x_ref, delta_ref, dA_cumsum_f_ref, dA_cumsum_b_ref

@pytest.fixture
def chunk_state_compare(create_chunk_state_scan_tensors):
    B, x, delta, dA_cumsum_f, dA_cumsum_b = create_chunk_state_scan_tensors
    B_ref, x_ref, delta_ref, dA_cumsum_f_ref, dA_cumsum_b_ref = create_ref_tensors(*create_chunk_state_scan_tensors)

    states_f = _chunk_state_fwd(B, x, delta, dA_cumsum_f)
    states_b = _chunk_state_fwd(B, x, delta, dA_cumsum_b, reverse=True)

    states_f_ref, states_b_ref = _chunk_state_fwd_bi(B_ref, x_ref, delta_ref, dA_cumsum_f_ref, dA_cumsum_b_ref)

    rtol, atol = (6e-4, 2e-3)

    assert_close(states_f, states_f_ref, rtol=rtol, atol=atol)
    assert_close(states_b, states_b_ref, rtol=rtol, atol=atol)

    return True
