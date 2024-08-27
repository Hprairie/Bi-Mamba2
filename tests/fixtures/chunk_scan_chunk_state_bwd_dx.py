# Copyright (c) 2024, Hayden Prairie.

import pytest

import torch
from torch.testing import assert_close

import triton

from ssd.uni.ssd_combined import _chunk_scan_chunk_state_bwd_dx
from ssd.bi.ssd_combined import _chunk_scan_chunk_state_bwd_dx as _chunk_scan_chunk_state_bwd_dx_bi

@pytest.fixture
def create_chunk_scan_bwd_dx_tensors(batch, seqlen, nheads, chunk_size, ngroups, headdim, dstate, dtype):
    torch.random.manual_seed(0)
    device = torch.device("cuda")
    assert nheads % ngroups == 0
    nchunks = triton.cdiv(seqlen, chunk_size)
    x = torch.randn(batch, seqlen, nheads, headdim).to(device=device, dtype=torch.float32).requires_grad_()
    dt = torch.randn(batch, nheads, nchunks, chunk_size).to(device=device, dtype=torch.float32).requires_grad_()
    dA_cumsum_f = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    dA_cumsum_b = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    B = (-0.5 * torch.randn(batch, seqlen, ngroups, dstate).abs()).to(device=device, dtype=torch.float32).requires_grad_()
    CB = (-0.5 * torch.randn(batch, nchunks, ngroups, chunk_size, chunk_size).abs()).to(device=device, dtype=torch.float32).requires_grad_()
    dout = (0.5 * torch.randn(batch, seqlen, nheads, headdim)).to(device=device, dtype=torch.float32).requires_grad_()
    dstates_f = (-0.5 * torch.randn(batch, nchunks, nheads, headdim, dstate).abs()).to(device=device, dtype=torch.float32).requires_grad_()
    dstates_b = (-0.5 * torch.randn(batch, nchunks, nheads, headdim, dstate).abs()).to(device=device, dtype=torch.float32).requires_grad_()

    return (
        x,
        dt,
        dA_cumsum_f,
        dA_cumsum_b,
        B,
        CB,
        dout,
        dstates_f,
        dstates_b,
    )

def create_ref_tensors(
        x,
        dt,
        dA_cumsum_f,
        dA_cumsum_b,
        B,
        CB,
        dout,
        dstates_f,
        dstates_b,
):
    x_ref = x.detach().clone().requires_grad_()
    dt_ref = dt.detach().clone().requires_grad_()
    dA_cumsum_f_ref = dA_cumsum_f.detach().clone().requires_grad_()
    dA_cumsum_b_ref = dA_cumsum_b.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    CB_ref = CB.detach().clone().requires_grad_()
    dout_ref = dout.detach().clone().requires_grad_()
    dstates_f_ref = dstates_f.detach().clone().requires_grad_()
    dstates_b_ref = dstates_b.detach().clone().requires_grad_()

    return x_ref, dt_ref, dA_cumsum_f_ref, dA_cumsum_b_ref, B_ref, CB_ref, dout_ref, dstates_f_ref, dstates_b_ref


@pytest.fixture
def chunk_scan_bwd_dx_compare(create_chunk_scan_bwd_dx_tensors):
    x, dt, dA_cumsum_f, dA_cumsum_b, B, CB, dout, dstates_f, dstates_b = create_chunk_scan_bwd_dx_tensors
    x_ref, dt_ref, dA_cumsum_f_ref, dA_cumsum_b_ref, B_ref, CB_ref, dout_ref, dstates_f_ref, dstates_b_ref = create_ref_tensors(*create_chunk_scan_bwd_dx_tensors)

    dx, ddt, dD = _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum_f, B, CB, dout, dstates_f)
    dx_b, ddt_b, dD_b = _chunk_scan_chunk_state_bwd_dx(x.flip([1]), dt.flip([2, 3]), dA_cumsum_b.flip([2, 3]), B.flip([1]), CB.flip([1, 3, 4]), dout.flip([1]), dstates_b.flip([1]))

    dx_b = dx_b.flip([1])
    ddt_b = ddt_b.flip([2, 3])

    dx_ref, ddt_ref, dD_ref = _chunk_scan_chunk_state_bwd_dx_bi(x_ref, dt_ref, dA_cumsum_f_ref, dA_cumsum_b_ref, B_ref, CB_ref, dout_ref, dstates_f_ref, dstates_b_ref)

    rtol, atol = (6e-4, 2e-3)

    print(dx)
    print(dx_b)
    print(dx + dx_b)
    print(dx_ref)

    assert_close(dx + dx_b, dx_ref, rtol=rtol, atol=atol)
    assert_close(ddt + ddt_b, ddt_ref, rtol=rtol, atol=atol)

    return True
