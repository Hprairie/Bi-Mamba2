import torch

import triton
import triton.language as tl

from ssd.uni.ssd_combined import _mamba_chunk_scan_combined_fwd
from ssd.bi.ssd_combined import _mamba_chunk_scan_combined_fwd as _mamba_chunk_scan_combined_fwd_bi

def init(seqlen):
    batch = 2
    nheads = 4
    headdim = 16
    ngroups = 4
    dstate = 16
    chunk_size = 32
    delta_softplus = True
    device = torch.device("cuda")
    x = torch.randn((batch, seqlen, nheads, headdim)).to(device)
    dt = torch.randn((batch, seqlen, nheads)).to(device)
    A = torch.randn((nheads,)).to(device)
    B = torch.randn((batch, seqlen, ngroups, dstate)).to(device)
    C = torch.randn((batch, seqlen, ngroups, dstate)).to(device)
    D = torch.randn((nheads, dstate)).to(device)
    z = torch.randn((batch, seqlen, nheads, headdim)).to(device)
    delta_bias = torch.randn((nheads,)).to(device)
    return x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus

def uni_fwd(x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus):
    _, out_x, _, _, _, _ = _mamba_chunk_scan_combined_fwd(
            x=x, dt=dt, A=A, B=B, C=C, D=D, dt_bias=delta_bias, z=z, chunk_size=chunk_size, dt_softplus=delta_softplus,
            )
    return out_x

def bi_fwd(x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus):
    out, out_x, _, _, _, _, _, _, _ = _mamba_chunk_scan_combined_fwd_bi(x=x, dt=dt, A=A, B=B, C=C, D=D, dt_bias=delta_bias, z=z, chunk_size=chunk_size, dt_softplus=delta_softplus)
    return out_x

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['seqlen'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(1, 12, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['Causal Mamba2', 'Bi-Mamba2'],  # Possible values for `line_arg`.
        line_names=['Causal Mamba2', 'Bi-Mamba2'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='ms',  # Label name for the y-axis.
        plot_name='Bidirectional Mamba Performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(seqlen, provider):
    x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus = init(seqlen)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'Causal Mamba2':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: uni_fwd(x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus), quantiles=quantiles, rep=2000, warmup=500)
    if provider == 'Bi-Mamba2':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: bi_fwd(x, dt, A, B, C, D, delta_bias, z, chunk_size, delta_softplus), quantiles=quantiles, rep=2000, warmup=500)
    return ms, max_ms, min_ms
    gbps = lambda ms: 3 * exp.numel() * exp.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True)
