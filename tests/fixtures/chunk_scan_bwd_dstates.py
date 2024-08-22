# Copyright (c) 2024, Hayden Prairie.

import pytest

import torch
from torch.testing import assert_close

import triton

from ssd.uni.ssd_chunk_scan import _chunk_scan_bwd_dstates
from ssd.bi.ssd_chunk_scan import _chunk_scan_bwd_dstates as _chunk_scan_bwd_dstates_bi

@pytest.fixture
def create_chunk_state_scan_bwd_dstate_tensors(batch, seqlen, nheads, chunk_size, ngroups, headdim, dstate, dtype):
    torch.random.manual_seed(0)
    device = torch.device("cuda")
    assert nheads % ngroups == 0
    nchunks = triton.cdiv(seqlen, chunk_size)
    dA_cumsum_f = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    dA_cumsum_b = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    C = (-0.5 * torch.randn(batch, seqlen, ngroups, dstate).abs()).to(device=device, dtype=torch.float32).requires_grad_()
    dout = (0.5 * torch.randn(batch, seqlen, nheads, headdim)).to(device=device, dtype=torch.float32).requires_grad_()

    return (
            C,
            dout,
            dA_cumsum_f,
            dA_cumsum_b,
    )

def create_ref_tensors(
    C,
    dout,
    dA_cumsum_f,
    dA_cumsum_b,
):
    C_ref = C.detach().clone().requires_grad_()
    dout_ref = dout.detach().clone().requires_grad_()
    dA_cumsum_f_ref = dA_cumsum_f.detach().clone().requires_grad_()
    dA_cumsum_b_ref = dA_cumsum_b.detach().clone().requires_grad_()

    return C_ref, dout_ref, dA_cumsum_f_ref, dA_cumsum_b_ref

@pytest.fixture
def chunk_scan_bwd_dstate_compare(create_chunk_state_scan_bwd_dstate_tensors):
    C, dout, dA_cumsum_f, dA_cumsum_b = create_chunk_state_scan_bwd_dstate_tensors
    C_ref, dout_ref, dA_cumsum_f_ref, dA_cumsum_b_ref = create_ref_tensors(*create_chunk_state_scan_bwd_dstate_tensors)

    dstates_f = _chunk_scan_bwd_dstates(C, dA_cumsum_f, dout)
    dstates_b = _chunk_scan_bwd_dstates(C, dA_cumsum_b, dout)

    dstates_f_ref, dstates_b_ref = _chunk_scan_bwd_dstates_bi(C_ref, dA_cumsum_f_ref, dA_cumsum_b_ref, dout)

    rtol, atol = (6e-4, 2e-3)

    assert_close(dstates_f, dstates_f_ref, rtol=rtol, atol=atol)
    assert_close(dstates_b, dstates_b_ref, rtol=rtol, atol=atol)

    return True
