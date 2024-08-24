# Copyright (c) 2024, Hayden Prairie.

import pytest

import torch
from torch.testing import assert_close

import triton

from ssd.uni.ssd_chunk_scan import _chunk_scan_bwd_ddAcs_stable
from ssd.bi.ssd_chunk_scan import _chunk_scan_bwd_ddAcs_stable_bwd as _chunk_scan_bwd_ddAcs_stable_bwd

@pytest.fixture
def create_chunk_scan_bwd_ddAcs_tensors(batch, seqlen, nheads, chunk_size, ngroups, headdim, dstate, dtype):
    torch.random.manual_seed(0)
    device = torch.device("cuda")
    assert nheads % ngroups == 0
    nchunks = triton.cdiv(seqlen, chunk_size)
    x = torch.randn(batch, seqlen, nheads, headdim).to(device=device, dtype=torch.float32).requires_grad_()
    dt = torch.randn(batch, nheads, nchunks, chunk_size).to(device=device, dtype=torch.float32).requires_grad_()
    dA_cumsum_b = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    CB = (-0.5 * torch.randn(batch, nchunks, ngroups, chunk_size, chunk_size).abs()).to(device=device, dtype=torch.float32).requires_grad_()
    dout = (0.5 * torch.randn(batch, seqlen, nheads, headdim)).to(device=device, dtype=torch.float32).requires_grad_()

    return (
        x,
        dt,
        dA_cumsum_b,
        CB,
        dout,
    )

def create_ref_tensors(
        x,
        dt,
        dA_cumsum_b,
        CB,
        dout,
):
    x_ref = x.detach().clone().requires_grad_()
    dt_ref = dt.detach().clone().requires_grad_()
    dA_cumsum_b_ref = dA_cumsum_b.detach().clone().requires_grad_()
    CB_ref = CB.detach().clone().requires_grad_()
    dout_ref = dout.detach().clone().requires_grad_()

    return x_ref, dt_ref, dA_cumsum_b_ref, CB_ref, dout_ref


@pytest.fixture
def chunk_scan_bwd_ddAcs_stable_compare(create_chunk_scan_bwd_ddAcs_tensors):
    x, dt, dA_cumsum_b, CB, dout = create_chunk_scan_bwd_ddAcs_tensors 
    x_ref, dt_ref, dA_cumsum_b_ref, CB_ref, dout_ref = create_ref_tensors(*create_chunk_scan_bwd_ddAcs_tensors)


    ddA_b = _chunk_scan_bwd_ddAcs_stable(x.flip([1]), dt.flip([2, 3]), dA_cumsum_b.flip([2, 3]), dout.flip([1]), CB.flip([3, 4]))
    ddA_b = ddA_b.flip([2, 3])

    ddA_b_ref = _chunk_scan_bwd_ddAcs_stable_bwd(x_ref, dt_ref, dA_cumsum_b_ref, dout_ref, CB_ref)

    # print(ddA_b)
    # print(ddA_b_ref)

    rtol, atol = (6e-4, 2e-3)

    assert_close(ddA_b, ddA_b_ref, rtol=rtol, atol=atol)

    return True
