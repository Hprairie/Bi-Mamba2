# Copyright (c) 2024, Hayden Prairie.

import pytest

import torch
from torch.testing import assert_close

import triton

from ssd.uni.ssd_chunk_scan import _chunk_scan_bwd_dC
from ssd.bi.ssd_chunk_scan import _chunk_scan_bwd_dC as _chunk_scan_bwd_dC_bi

@pytest.fixture
def create_chunk_scan_bwd_dc_tensors(batch, seqlen, nheads, chunk_size, ngroups, headdim, dstate, dtype):
    torch.random.manual_seed(0)
    device = torch.device("cuda")
    assert nheads % ngroups == 0
    nchunks = triton.cdiv(seqlen, chunk_size)
    dA_cumsum_f = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    dA_cumsum_b = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    C = (-0.5 * torch.randn(batch, seqlen, ngroups, dstate).abs()).to(device=device, dtype=torch.float32).requires_grad_()
    dout = (0.5 * torch.randn(batch, seqlen, nheads, headdim)).to(device=device, dtype=torch.float32).requires_grad_()
    prev_states_f = (-0.5 * torch.randn(batch, nchunks, nheads, headdim, dstate).abs()).to(device=device, dtype=torch.float32).requires_grad_()
    prev_states_b = (-0.5 * torch.randn(batch, nchunks, nheads, headdim, dstate).abs()).to(device=device, dtype=torch.float32).requires_grad_()

    return (
        prev_states_f,
        prev_states_b,
        dA_cumsum_f,
        dA_cumsum_b,
        dout,
        C,
    )

def create_ref_tensors(
    prev_states_f,
    prev_states_b,
    dA_cumsum_f,
    dA_cumsum_b,
    dout,
    C,
):
    dout_ref = dout.detach().clone().requires_grad_()
    dA_cumsum_f_ref = dA_cumsum_f.detach().clone().requires_grad_()
    dA_cumsum_b_ref = dA_cumsum_b.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    prev_states_f_ref = prev_states_f.detach().clone().requires_grad_()
    prev_states_b_ref = prev_states_b.detach().clone().requires_grad_()

    return prev_states_f_ref, prev_states_b_ref, dA_cumsum_f_ref, dA_cumsum_b_ref, dout_ref, C_ref


@pytest.fixture
def chunk_scan_bwd_dc_compare(create_chunk_scan_bwd_dc_tensors):
    prev_states_f, prev_states_b, dA_cumsum_f, dA_cumsum_b, dout, C = create_chunk_scan_bwd_dc_tensors
    prev_states_f_ref, prev_states_b_ref, dA_cumsum_f_ref, dA_cumsum_b_ref, dout_ref, C_ref = create_ref_tensors(*create_chunk_scan_bwd_dc_tensors)

    dC_f, ddA_cumsum_f = _chunk_scan_bwd_dC(prev_states_f, dA_cumsum_f, dout, C=C)
    dC_b, ddA_cumsum_b = _chunk_scan_bwd_dC(prev_states_b.flip([1]), dA_cumsum_b.flip([2, 3]), dout.flip([1]), C=C.flip([1]))

    dC_b = dC_b.flip([1])
    ddA_cumsum_b = ddA_cumsum_b.flip([2, 3])

    dC_ref, ddA_cumsum_f_ref, ddA_cumsum_b_ref = _chunk_scan_bwd_dC_bi(prev_states_f_ref, prev_states_b_ref, dA_cumsum_f_ref, dA_cumsum_b_ref, dout_ref, C=C_ref)

    rtol, atol = (6e-4, 2e-3)

    # print(ddA_cumsum_b)
    # print(ddA_cumsum_b_ref)

    assert_close(dC_ref, dC_b +  dC_f, rtol=rtol, atol=atol)
    assert_close(ddA_cumsum_f, ddA_cumsum_f_ref, rtol=rtol, atol=atol)
    assert_close(ddA_cumsum_b, ddA_cumsum_b_ref, rtol=rtol, atol=atol)

    return True
