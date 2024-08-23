# Copyright (c) 2024, Hayden Prairie.

import pytest

import torch
from torch.testing import assert_close

import triton

from ssd.uni.ssd_chunk_state import _chunk_state_bwd_db 
from ssd.bi.ssd_chunk_state import _chunk_state_bwd_db as _chunk_state_bwd_db_bi

@pytest.fixture
def create_chunk_state_bwd_db_tensors(batch, seqlen, nheads, chunk_size, ngroups, headdim, dstate, dtype):
    torch.random.manual_seed(0)
    device = torch.device("cuda")
    assert nheads % ngroups == 0
    nchunks = triton.cdiv(seqlen, chunk_size)
    x = (0.5 * torch.randn(batch, seqlen, nheads, headdim)).to(device=device, dtype=torch.float32).requires_grad_()
    dt = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    dA_cumsum_f = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    dA_cumsum_b = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    B = (-0.5 * torch.randn(batch, seqlen, ngroups, dstate).abs()).to(device=device, dtype=torch.float32).requires_grad_()
    dstates_f = (-0.5 * torch.randn(batch, nchunks, nheads, headdim, dstate).abs()).to(device=device, dtype=torch.float32).requires_grad_()
    dstates_b = (-0.5 * torch.randn(batch, nchunks, nheads, headdim, dstate).abs()).to(device=device, dtype=torch.float32).requires_grad_()

    return (
            x,
            dt,
            dA_cumsum_f,
            dA_cumsum_b,
            B,
            dstates_f,
            dstates_b,
    )

def create_ref_tensors(
    x,
    dt,
    dA_cumsum_f,
    dA_cumsum_b,
    B,
    dstates_f,
    dstates_b,
):
    x_ref = x.detach().clone().requires_grad_()
    dt_ref = dt.detach().clone().requires_grad_()
    dA_cumsum_f_ref = dA_cumsum_f.detach().clone().requires_grad_()
    dA_cumsum_b_ref = dA_cumsum_b.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    dstates_f_ref = dstates_f.detach().clone().requires_grad_()
    dstates_b_ref = dstates_b.detach().clone().requires_grad_()

    return x_ref, dt_ref, dA_cumsum_f_ref, dA_cumsum_b_ref, B_ref, dstates_f_ref, dstates_b_ref


@pytest.fixture
def chunk_state_bwd_db_compare(create_chunk_state_bwd_db_tensors):
    x, dt, dA_cumsum_f, dA_cumsum_b, B, dstates_f, dstates_b = create_chunk_state_bwd_db_tensors
    x_ref, dt_ref, dA_cumsum_f_ref, dA_cumsum_b_ref, B_ref, dstates_f_ref, dstates_b_ref = create_ref_tensors(*create_chunk_state_bwd_db_tensors)

    db_f, ddA_cumsum_f = _chunk_state_bwd_db(x, dt, dA_cumsum_f, dstates_f, B=B)
    db_b, ddA_cumsum_b = _chunk_state_bwd_db(x.flip([1]), dt.flip([2, 3]), dA_cumsum_b.flip([2, 3]), dstates_b.flip([1]), B=B.flip([1]))
    db_b = db_b.flip([1])
    ddA_cumsum_b = ddA_cumsum_b.flip([2, 3])

    
    db_ref, ddA_cumsum_f_ref, ddA_cumsum_b_ref = _chunk_state_bwd_db_bi(x_ref, dt_ref, dA_cumsum_f_ref, dA_cumsum_b_ref, dstates_f_ref, dstates_b_ref, B=B_ref)

    rtol, atol = (6e-4, 2e-3)

    # print(db_f)
    # print(db_b)
    # print(db_f + db_b)
    # print(db_ref)
    # print(ddA_cumsum_b)
    # print(ddA_cumsum_b_ref)

    assert_close(db_ref, db_b + db_f, rtol=rtol, atol=atol)
    assert_close(ddA_cumsum_f, ddA_cumsum_f_ref, rtol=rtol, atol=atol)
    assert_close(ddA_cumsum_b, ddA_cumsum_b_ref, rtol=rtol, atol=atol)

    return True
