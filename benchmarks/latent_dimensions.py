"""Loop/batch/chunk propagation benchmark (no CI timing assertions).

Run with python benchmarks/latent_dimensions.py --output results.json.
Default matrix includes 128, 256, and a 400-pixel HARMONI-sized grid. Atmosphere
sampling is excluded. Forward+backward retains the complete chunk graph.
"""

import argparse
import json
import platform
from pathlib import Path
import torch
from torch.utils.benchmark import Timer
from fiatlux import (
    Field,
    FieldDimension,
    Grid,
    Spectrum,
    PhotometricBand,
    FFTPropagator,
    MFTPropagator,
)


def benchmark(args):
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    rows = []
    for size in args.sizes:
        grid = Grid(size, size, 0.02, 0.02, dtype=dtype, device=device)
        for method in args.methods:
            for nw in ([1] if method == "fft" else args.wavelengths):
                spectrum = Spectrum(
                    0, PhotometricBand.HCM2, nw, device=device, dtype=dtype
                )
                for regime in args.regimes:
                    kw = (
                        dict(focal_length=2.0)
                        if regime == "fraunhofer"
                        else dict(propagation="fresnel", distance=2.0)
                    )
                    if method == "fft":
                        propagator = FFTPropagator(**kw)
                    else:
                        propagator = MFTPropagator(
                            output_grid=Grid(
                                size, size, 1e-6, 1e-6, device=device, dtype=dtype
                            ),
                            **kw,
                        )
                    for count in args.times:
                        rng = torch.Generator(device=device).manual_seed(812)
                        # A frozen input cube; no sampling/OPD construction in timed calls.
                        real = torch.randn(
                            count,
                            nw,
                            size,
                            size,
                            generator=rng,
                            device=device,
                            dtype=dtype,
                        )
                        amplitude = torch.complex(real, real.sin())

                        def propagate(chunk, labelled=True):
                            dims = (
                                (FieldDimension("time", len(chunk)),)
                                if labelled
                                else ()
                            )
                            return (
                                propagator.apply(Field(chunk, grid, spectrum, dims))
                                .intensity()
                                .sum(-3)
                            )

                        with torch.no_grad():
                            reference = torch.stack(
                                [propagate(v, False) for v in amplitude]
                            ).sum(0)
                        for backward in (False, True):
                            amplitude.requires_grad_(backward)
                            modes = [("loop", 1), ("batch", count)] + [
                                ("chunk", n) for n in args.chunks if n < count
                            ]
                            baseline = None
                            for mode, chunk_size in modes:

                                def run():
                                    with torch.set_grad_enabled(backward):
                                        total = None
                                        for start in range(0, count, chunk_size):
                                            chunk = amplitude[
                                                start : start + chunk_size
                                            ]
                                            image = (
                                                propagate(chunk[0], False)
                                                if mode == "loop"
                                                else propagate(chunk).sum(0)
                                            )
                                            total = (
                                                image
                                                if total is None
                                                else total + image
                                            )
                                        if backward:
                                            torch.autograd.grad(total.sum(), amplitude)
                                        return total.detach()

                                try:
                                    for _ in range(2):
                                        actual = run()
                                    if device.type == "cuda":
                                        torch.cuda.synchronize()
                                        torch.cuda.reset_peak_memory_stats()
                                        begin, end = torch.cuda.Event(
                                            enable_timing=True
                                        ), torch.cuda.Event(enable_timing=True)
                                        begin.record()
                                        for _ in range(args.repeats):
                                            run()
                                        end.record()
                                        end.synchronize()
                                        seconds = (
                                            begin.elapsed_time(end)
                                            / 1000
                                            / args.repeats
                                        )
                                        allocated = torch.cuda.max_memory_allocated()
                                        reserved = torch.cuda.max_memory_reserved()
                                    else:
                                        seconds = (
                                            Timer(
                                                stmt="run()",
                                                globals={"run": run},
                                                num_threads=args.threads,
                                            )
                                            .blocked_autorange(
                                                min_run_time=args.min_time
                                            )
                                            .median
                                        )
                                        allocated = reserved = None
                                    delta = (actual - reference).abs()
                                    if mode == "loop":
                                        baseline = seconds
                                    row = dict(
                                        size=size,
                                        time_samples=count,
                                        wavelengths=nw,
                                        method=method,
                                        regime=regime,
                                        backward=backward,
                                        mode=mode,
                                        chunk_size=chunk_size,
                                        seconds=seconds,
                                        samples_per_second=count / seconds,
                                        speedup=baseline / seconds,
                                        max_absolute_error=float(delta.max()),
                                        max_relative_error=float(
                                            (
                                                delta
                                                / reference.abs().clamp_min(
                                                    torch.finfo(dtype).tiny
                                                )
                                            ).max()
                                        ),
                                        peak_cuda_allocated=allocated,
                                        peak_cuda_reserved=reserved,
                                    )
                                except torch.cuda.OutOfMemoryError:
                                    torch.cuda.empty_cache()
                                    row = dict(
                                        size=size,
                                        time_samples=count,
                                        wavelengths=nw,
                                        method=method,
                                        regime=regime,
                                        backward=backward,
                                        mode=mode,
                                        chunk_size=chunk_size,
                                        error="CUDA out of memory",
                                    )
                                rows.append(row)
                                print(json.dumps(row), flush=True)
    return dict(
        environment=dict(
            torch=torch.__version__,
            platform=platform.platform(),
            device=str(device),
            dtype=args.dtype,
            threads=args.threads,
        ),
        results=rows,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 400])
    parser.add_argument("--times", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    parser.add_argument("--wavelengths", type=int, nargs="+", default=[1, 3])
    parser.add_argument(
        "--methods", nargs="+", choices=["fft", "mft"], default=["fft", "mft"]
    )
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=["fraunhofer", "fresnel"],
        default=["fraunhofer", "fresnel"],
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-time", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("latent-benchmark.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(benchmark(args), indent=2))
