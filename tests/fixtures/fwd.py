# Copyright (c) 2024, Hayden Prairie.

import pytest

import torch
from torch.testing import assert_close

import triton

from ssd.uni.ssd_combined import _mamba_chunk_scan_combined_fwd
from ssd.bi.ssd_combined import _mamba_chunk_scan_combined_fwd as _mamba_chunk_scan_combined_fwd_bi

@pytest.fixture
def create_fwd_scan_tensors(batch, seqlen, nheads, ngroups, chunk_size, headdim, dstate, delta_softplus, dtype):
    torch.random.manual_seed(0)
    device = torch.device("cuda")
    device = torch.device("cuda")
    x = torch.randn((batch, seqlen, nheads, headdim)).to(device, dtype=dtype)
    dt = torch.randn((batch, seqlen, nheads)).to(device, dtype=dtype)
    A = torch.randn((nheads,)).to(device, dtype=dtype)
    B = torch.randn((batch, seqlen, ngroups, dstate)).to(device, dtype=dtype)
    C = torch.randn((batch, seqlen, ngroups, dstate)).to(device, dtype=dtype)
    D = torch.randn((nheads, dstate)).to(device, dtype=dtype)
    z = torch.randn((batch, seqlen, nheads, headdim)).to(device, dtype=dtype)
    delta_bias = torch.randn((nheads,)).to(device, dtype=dtype)
    z = None
    D = None
    return x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus

def create_ref_tensors(
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
    return x_ref, dt_ref, A_ref, B_ref, C_ref, D_ref, delta_bias_ref, z_ref, chunk_size, delta_softplus


@pytest.fixture
def fwd_compare(create_fwd_scan_tensors):
    x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus = create_fwd_scan_tensors
    x_ref, dt_ref, A_ref, B_ref, C_ref, D_ref, delta_bias_ref, z_ref, chunk_size, delta_softplus = create_ref_tensors(*create_fwd_scan_tensors)

    out, out_x, dt_f, dA_cumsum_f, states_f, final_states_f = _mamba_chunk_scan_combined_fwd(
            x=x, dt=dt, A=A, B=B, C=C, D=D, dt_bias=delta_bias, z=z, chunk_size=chunk_size, dt_softplus=delta_softplus,
            )
    z = z.flip([1]) if z is not None else None
    out, out_x2, dt_b, dA_cumsum_b, states_b, final_states_b = _mamba_chunk_scan_combined_fwd(
            x=x.flip([1]), dt=dt.flip([1]), A=A, B=B.flip([1]), C=C.flip([1]), D=None, dt_bias=delta_bias, z=z, chunk_size=chunk_size, dt_softplus=delta_softplus,
            )
    if z is not None:
        out_x = out_x + out_x2.flip([1])
    else:
        out = out + out.flip([1])

    out_ref, out_x_ref, dt_ref, dA_cumsum_f_ref, dA_cumsum_b_ref, states_f_ref, states_b_ref, final_states_f_ref, final_states_b_ref = _mamba_chunk_scan_combined_fwd_bi(x=x_ref, dt=dt_ref, A=A_ref, B=B_ref, C=C_ref, D=D_ref, dt_bias=delta_bias_ref, z=z_ref, chunk_size=chunk_size, dt_softplus=delta_softplus)

    print(f'{dA_cumsum_f=}')
    print(f'{dA_cumsum_f_ref=}')
    print(f'{dA_cumsum_b=}')
    print(f'{dA_cumsum_b_ref=}')
    print(f'{states_f=}')
    print(f'{states_f_ref=}')
    print(f'{states_b=}')
    print(f'{states_b_ref=}')
    print(f'{final_states_f=}')
    print(f'{final_states_f_ref=}')
    print(f'{final_states_b=}')
    print(f'{final_states_b_ref=}')
    print(f'{out=}')
    print(f'{out_ref=}')
    print(f'{out_x=}')
    print(f'{out_x_ref=}')

    rtol, atol = (6e-4, 2e-3)

    if z is not None:
        assert_close(out_x, out_x_ref, rtol=rtol, atol=atol)
    else:
        assert_close(out, out_ref, rtol=rtol, atol=atol)

    return True
