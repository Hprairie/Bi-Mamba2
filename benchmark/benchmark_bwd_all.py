import torch

import triton
import triton.language as tl

from ssd.uni.ssd_combined import _mamba_chunk_scan_combined_bwd
from ssd.bi.ssd_combined import _mamba_chunk_scan_combined_bwd as _mamba_chunk_scan_combined_bwd_bi

def init(seqlen):
    batch = 2
    nheads = 4
    headdim = 16
    ngroups = 4
    dstate = 16
    chunk_size = 128
    delta_softplus = True
    device = torch.device("cuda")
    x = torch.randn((batch, seqlen, nheads, headdim)).to(device)
    dout = torch.rand_like(x)
    out = torch.rand_like(x)
    dt = torch.randn((batch, seqlen, nheads)).to(device)
    A = torch.randn((nheads,)).to(device)
    B = torch.randn((batch, seqlen, ngroups, dstate)).to(device)
    C = torch.randn((batch, seqlen, ngroups, dstate)).to(device)
    D = torch.randn((nheads, dstate)).to(device)
    z = torch.randn((batch, seqlen, nheads, headdim)).to(device)
    delta_bias = torch.randn((nheads,)).to(device)
    return dout, out, x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus

def uni_fwd(dout, out, x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus):
    dx, ddt, dA, dB, dC, dD, dz, ddt_bias, _ = _mamba_chunk_scan_combined_bwd(
            dout=dout, out=out, x=x, dt=dt, A=A, B=B, C=C, D=D, dt_bias=delta_bias, z=z, chunk_size=chunk_size, dt_softplus=delta_softplus,
            )
    dx2, ddt2, dA2, dB2, dC2, dD2, dz2, ddt_bias2, _ = _mamba_chunk_scan_combined_bwd(
            dout=dout.flip([1]), out=out.flip([1]), x=x.flip([1]), dt=dt.flip([1]), A=A, B=B.flip([1]), C=C.flip([1]), D=D, dt_bias=delta_bias, z=z.flip([1]), chunk_size=chunk_size, dt_softplus=delta_softplus,
            )
    return dx + dx2.flip([1]), ddt + ddt2.flip([1]), dA + dA2, dB + dB2.flip([1]), dC + dC2.flip([1]), dD + dD2, dz + dz2.flip([1]), ddt_bias + ddt_bias2

def causal_fwd(dout, out, x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus):
    dx, ddt, dA, dB, dC, dD, dz, ddt_bias, _ = _mamba_chunk_scan_combined_bwd(
            dout=dout, out=out, x=x, dt=dt, A=A, B=B, C=C, D=D, dt_bias=delta_bias, z=z, chunk_size=chunk_size, dt_softplus=delta_softplus,
            )
    return dx, ddt, dA, dB, dC, dD, dz, ddt_bias

def bi_fwd(dout, out, x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus):
    dx, ddt, dA, dB, dC, dD, dz, ddt_bias = _mamba_chunk_scan_combined_bwd_bi(dout=dout, out=out, x=x, dt=dt, A=A, B=B, C=C, D=D, dt_bias=delta_bias, z=z, chunk_size=chunk_size, dt_softplus=delta_softplus)
    return dx, ddt, dA, dB, dC, dD, dz, ddt_bias

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seqlen'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(1, 14, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['Naive Mamba2', 'Bi-Mamba2', 'Causal Mamba2'],  # Possible values for `line_arg`.
        line_names=['Naive Mamba2', 'Bi-Mamba2', 'Causal Mamba2'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # Line styles.
        ylabel='ms',  # Label name for the y-axis.
        plot_name='Bi-Directional Bwd Pass Mamba Performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(seqlen, provider):
    dout, out, x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus = init(seqlen)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'Naive Mamba2':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: uni_fwd(dout, out, x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus), quantiles=quantiles, rep=2000, warmup=500)
    if provider == 'Bi-Mamba2':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: bi_fwd(dout, out, x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus), quantiles=quantiles, rep=2000, warmup=500)
    if provider == 'Causal Mamba2':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: causal_fwd(dout, out, x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus), quantiles=quantiles, rep=2000, warmup=500)
    return ms, max_ms, min_ms
    gbps = lambda ms: 3 * exp.numel() * exp.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True)
