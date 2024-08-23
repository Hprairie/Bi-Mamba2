# Copyright (c) 2024, Hayden Prairie.

import pytest

import torch
from torch.testing import assert_close

import triton

from ssd.uni.ssd_chunk_scan import _chunk_scan_bwd_dcb
from ssd.bi.ssd_chunk_scan import _chunk_scan_bwd_dcb as _chunk_scan_bwd_dcb_bi

@pytest.fixture
def create_chunk_state_bwd_dcb_tensors(batch, seqlen, nheads, chunk_size, ngroups, headdim, dstate, dtype):
    torch.random.manual_seed(0)
    device = torch.device("cuda")
    assert nheads % ngroups == 0
    nchunks = triton.cdiv(seqlen, chunk_size)
    x = (0.5 * torch.randn(batch, seqlen, nheads, headdim)).to(device=device, dtype=torch.float32).requires_grad_()
    dt = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    dA_cumsum_f = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    dA_cumsum_b = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    dout = (0.5 * torch.randn(batch, seqlen, nheads, headdim)).to(device=device, dtype=torch.float32).requires_grad_()

    return (
        x,
        dt,
        dA_cumsum_f,
        dA_cumsum_b,
        dout,
        ngroups
    )

def create_ref_tensors(
    x,
    dt,
    dA_cumsum_f,
    dA_cumsum_b,
    dout,
    ngroups
):
    x_ref = x.detach().clone().requires_grad_()
    dt_ref = dt.detach().clone().requires_grad_()
    dA_cumsum_f_ref = dA_cumsum_f.detach().clone().requires_grad_()
    dA_cumsum_b_ref = dA_cumsum_b.detach().clone().requires_grad_()
    dout_ref = dout.detach().clone().requires_grad_()

    return x_ref, dt_ref, dA_cumsum_f_ref, dA_cumsum_b_ref, dout_ref, ngroups


@pytest.fixture
def chunk_state_bwd_dcb_compare(create_chunk_state_bwd_dcb_tensors):
    x, dt, dA_cumsum_f, dA_cumsum_b, dout, ngroups = create_chunk_state_bwd_dcb_tensors
    x_ref, dt_ref, dA_cumsum_f_ref, dA_cumsum_b_ref, dout_ref, ngroups_ref = create_ref_tensors(*create_chunk_state_bwd_dcb_tensors)

    dCB_f = _chunk_scan_bwd_dcb(x, dt, dA_cumsum_f, dout, ngroups=ngroups) 
    dCB_b = _chunk_scan_bwd_dcb(x.flip([1]), dt.flip([2, 3]), dA_cumsum_b.flip([2, 3]), dout.flip([1]), ngroups=ngroups)
    dCB_b = dCB_b.flip([3, 4])

    dCB_ref = _chunk_scan_bwd_dcb_bi(x_ref, dt_ref, dA_cumsum_f_ref, dA_cumsum_b_ref, dout_ref, ngroups=ngroups_ref)

    # torch.set_printoptions(threshold=10000) # Set a large threshold
    # print(dCB_f + dCB_b)
    # print(dCB_ref)

    rtol, atol = (6e-4, 2e-3)

    assert_close(dCB_ref, dCB_f + dCB_b, rtol=rtol, atol=atol)

    return True
