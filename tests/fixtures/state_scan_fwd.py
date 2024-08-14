# Copyright (c) 2024, Hayden Prairie.

import pytest

import torch
from torch.testing import assert_close

import triton

from ssd.uni.ssd_chunk_scan import _chunk_scan_fwd
from ssd.bi.ssd_chunk_scan import _chunk_scan_fwd as _chunk_scan_fwd_bi

@pytest.fixture
def create_chunk_scan_tensors(batch, seqlen, nheads, ngroups, chunk_size, headdim, dstate, has_z, has_d, dtype):
    torch.random.manual_seed(0)
    device = torch.device("cuda")
    nchunks = triton.cdiv(seqlen, chunk_size)
    CB = (-0.5 * torch.randn(batch, nchunks, ngroups, chunk_size, chunk_size).abs()).to(device=device, dtype=torch.float32).requires_grad_()
    x = (0.5 * torch.randn(batch, seqlen, nheads, headdim)).to(device=device, dtype=torch.float32).requires_grad_()
    delta = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    dA_cumsum_f = (0.5 * torch.randn(batch, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    dA_cumsum_b = (0.5 * torch.randn(nheads, nheads, nchunks, chunk_size)).to(device=device, dtype=torch.float32).requires_grad_()
    C = (0.5 * torch.randn(batch, seqlen, ngroups, dstate)).to(device=device, dtype=torch.float32).requires_grad_()
    states_f = (0.5 * torch.randn(batch, nchunks, nheads, headdim, dstate)).to(device=device, dtype=torch.float32).requires_grad_()
    states_b = (0.5 * torch.randn(batch, nchunks, nheads, headdim, dstate)).to(device=device, dtype=torch.float32).requires_grad_()
    D = (0.5 * torch.randn(nheads, headdim)).to(device=device, dtype=torch.float32).requires_grad_() if has_d else None
    z = (0.5 * torch.randn(batch, seqlen, nheads, headdim)).to(device=device, dtype=torch.float32).requires_grad_() if has_z else None

    return (
        CB,
        x,
        delta,
        dA_cumsum_f,
        dA_cumsum_b,
        C,
        states_f,
        states_b,
        D,
        z,
    )

def create_ref_tensors(
    CB: torch.Tensor,
    x: torch.Tensor,
    delta: torch.Tensor,
    dA_cumsum_f: torch.Tensor,
    dA_cumsum_b: torch.Tensor,
    C: torch.Tensor,
    states_f: torch.Tensor,
    states_b: torch.Tensor,
    D: torch.Tensor,
    z: torch.Tensor,
):
    CB_ref = CB.detach().clone().requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    delta_ref = delta.detach().clone().requires_grad_()
    dA_cumsum_f_ref = dA_cumsum_f.detach().clone().requires_grad_()
    dA_cumsum_b_ref = dA_cumsum_b.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    states_f_ref = states_f.detach().clone().requires_grad_()
    states_b_ref = states_b.detach().clone().requires_grad_()
    D_ref = D.detach().clone().requires_grad_() if D is not None else None
    z_ref = z.detach().clone().requires_grad_() if z is not None else None

    return CB_ref, x_ref, delta_ref, dA_cumsum_f_ref, dA_cumsum_b_ref, C_ref, states_f_ref, states_b_ref, D_ref, z_ref


@pytest.fixture
def cumsum_compare(create_chunk_scan_tensors):
    CB, x, delta, dA_cumsum_f, dA_cumsum_b, C, states_f, states_b, D, z = create_chunk_scan_tensors
    CB_ref, x_ref, delta_ref, dA_cumsum_f_ref, dA_cumsum_b_ref, C_ref, states_f_ref, states_b_ref, D_ref, z_ref = create_ref_tensors(*create_chunk_scan_tensors)

    out, out_x = _chunk_scan_fwd(CB, x, delta, dA_cumsum_f, C, states_f, D, z)
    out2, out_x2 = _chunk_scan_fwd(CB.flip([1, 3, 4]), x.flip([1]), delta.flip([2, 3]), dA_cumsum_b.flip([2]), C.flip([1, 2]), states_b.flip([1, 3, 4]), D.flip([0]), z.flip([1]) if z is not None else None)

    rtol, atol = (6e-4, 2e-3)

    assert_close(dA_cusum_f, dA_cumsum_f_ref, rtol=rtol, atol=atol)
    assert_close(dt_out, dt_out, rtol=rtol, atol=atol)

    return True
