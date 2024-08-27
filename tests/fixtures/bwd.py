# Copyright (c) 2024, Hayden Prairie.

import pytest

import torch
from torch.testing import assert_close

from ssd.uni.ssd_combined import _mamba_chunk_scan_combined_bwd
from ssd.bi.ssd_combined import _mamba_chunk_scan_combined_bwd as _mamba_chunk_scan_combined_bwd_bi

@pytest.fixture
def create_bwd_scan_tensors(batch, seqlen, nheads, ngroups, chunk_size, headdim, dstate, delta_softplus, dtype):
    torch.random.manual_seed(0)
    device = torch.device("cuda")
    device = torch.device("cuda")
    x = torch.randn((batch, seqlen, nheads, headdim)).to(device, dtype=dtype)
    dt = (0.5 * torch.randn(batch, seqlen, nheads)).to(device, dtype=dtype)
    A =(-0.5 * torch.randn(nheads,).abs()).to(device, dtype=dtype)
    B = torch.randn((batch, seqlen, ngroups, dstate)).to(device, dtype=dtype)
    C = torch.randn((batch, seqlen, ngroups, dstate)).to(device, dtype=dtype)
    D = torch.randn((nheads, dstate)).to(device, dtype=dtype)
    z = torch.randn((batch, seqlen, nheads, headdim)).to(device, dtype=dtype)
    delta_bias = (0.5 * torch.randn((nheads,))).to(device, dtype=dtype)
    dout = torch.rand_like(x)
    out = torch.rand_like(x)
    z = None
    D = None
    return dout, out, x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus

def create_ref_tensors(
    dout: torch.Tensor,
    out: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    delta_bias: torch.Tensor,
    z: torch.Tensor,
    chunk_size: int,
    delta_softplus: bool,
):
    x_ref = x.detach().clone().requires_grad_()
    dt_ref = dt.detach().clone().requires_grad_()
    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    D_ref = D.detach().clone().requires_grad_() if D is not None else None
    delta_bias_ref = delta_bias.detach().clone().requires_grad_()
    z_ref = z.detach().clone().requires_grad_() if z is not None else None
    dout_ref = dout.detach().clone().requires_grad_()
    out_ref = out.detach().clone().requires_grad_()
    return dout_ref, out_ref, x_ref, dt_ref, A_ref, B_ref, C_ref, D_ref, delta_bias_ref, z_ref, chunk_size, delta_softplus


@pytest.fixture
def bwd_compare(create_bwd_scan_tensors):
    dout, out, x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus = create_bwd_scan_tensors
    dout_ref, out_ref, x_ref, dt_ref, A_ref, B_ref, C_ref, D_ref, delta_bias_ref, z_ref, chunk_size, delta_softplus = create_ref_tensors(*create_bwd_scan_tensors)

    dx, ddt, dA, dB, dC, dD, dz, ddt_bias, _  = _mamba_chunk_scan_combined_bwd(
            dout=dout, out=out, x=x, dt=dt, A=A, B=B, C=C, D=D, dt_bias=delta_bias, z=z, chunk_size=chunk_size, dt_softplus=delta_softplus,
            )
    z = z.flip([1]) if z is not None else None
    dx_2, ddt_2, dA_2, dB_2, dC_2, dD_2, dz_2, ddt_bias_2, _ = _mamba_chunk_scan_combined_bwd(
            dout=dout.flip([1]), out=out, x=x.flip([1]), dt=dt.flip([1]), A=A, B=B.flip([1]), C=C.flip([1]), D=None, dt_bias=delta_bias, z=z, chunk_size=chunk_size, dt_softplus=delta_softplus,
            )
    dx_ref, ddt_ref, dA_ref, dB_ref, dC_ref, dD_ref, dz_ref, ddt_bias_ref = _mamba_chunk_scan_combined_bwd_bi(dout=dout_ref, out=2*out_ref, x=x_ref, dt=dt_ref, A=A_ref, B=B_ref, C=C_ref, D=D_ref, dt_bias=delta_bias_ref, z=z_ref, chunk_size=chunk_size, dt_softplus=delta_softplus)

    rtol, atol = (6e-4, 2e-3)

    # print(ddt + ddt_2.flip([1]))
    # print(ddt_ref)
    assert_close(dx + dx_2.flip([1]), dx_ref, rtol=rtol, atol=atol)
    assert_close(dB + dB_2.flip([1]), dB_ref, rtol=rtol, atol=atol)
    assert_close(dC + dC_2.flip([1]), dC_ref, rtol=rtol, atol=atol)
    assert_close(dA + dA_2, dA_ref, rtol=rtol, atol=atol)
    assert_close(ddt + ddt_2.flip([1]), ddt_ref, rtol=rtol, atol=atol)
    if D is not None:
        assert_close(dD + dD_2, dD_ref, rtol=rtol, atol=atol)
    if z is not None:
        assert_close(dz + dz_2.flip([1]), dz_ref, rtol=rtol, atol=atol)
    if delta_bias is not None:
        assert_close(ddt_bias + ddt_bias_2, ddt_bias_ref, rtol=rtol, atol=atol)

    return True
