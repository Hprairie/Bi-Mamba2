# Copyright (c) 2024, Hayden Prairie.

import pytest

import torch
from torch.testing import assert_close

from ssd.uni.ssd_chunk_state import _chunk_cumsum_fwd
from ssd.bi.ssd_chunk_state import _chunk_cumsum_fwd as _chunk_cumsum_fwd_bi

@pytest.fixture
def create_chunk_cumsum_scan_tensors(batch, seqlen, nheads, chunk_size, has_delta_bias, softplus, dtype):
    torch.random.manual_seed(0)
    device = torch.device("cuda")
    A = (-0.5 * torch.randn(nheads,).abs()).to(device=device, dtype=torch.float32).requires_grad_()
    delta = (0.5 * torch.randn(batch, seqlen, nheads)).to(device=device, dtype=torch.float32).requires_grad_()
    delta_bias = (0.5 * torch.randn(nheads,)).to(device=device, dtype=torch.float32).requires_grad_() if has_delta_bias else None

    return (
        A,
        delta,
        delta_bias,
        chunk_size, 
        softplus,
        dtype
    )

def create_ref_tensors(
    A: torch.Tensor,
    delta: torch.Tensor,
    delta_bias: torch.Tensor,
    chunk_size: int,
    softplus: bool = False,
    dtype: torch.dtype = torch.float32,
):
    A_ref = A.detach().clone().requires_grad_()
    delta_ref = delta.detach().clone().requires_grad_()
    delta_bias_ref = delta_bias.detach().clone().requires_grad_() if delta_bias is not None else None

    return A_ref, delta_ref, delta_bias_ref

@pytest.fixture
def cumsum_compare(create_chunk_cumsum_scan_tensors):
    A, delta, delta_bias, chunk_size, delta_softplus, _ = create_chunk_cumsum_scan_tensors
    A_ref, delta_ref, delta_bias_ref = create_ref_tensors(*create_chunk_cumsum_scan_tensors)

    dA_cusum_f, dt_out = _chunk_cumsum_fwd(delta, A, chunk_size, delta_bias, delta_softplus)

    dA_cumsum_f_ref, dA_cumsum_b_ref, dt_out = _chunk_cumsum_fwd_bi(delta_ref, A_ref, chunk_size, delta_bias_ref, delta_softplus)

    rtol, atol = (6e-4, 2e-3)

    assert_close(dA_cusum_f, dA_cumsum_f_ref, rtol=rtol, atol=atol)
    assert_close(dt_out, dt_out, rtol=rtol, atol=atol)

    return True
