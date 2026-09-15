"""Physical equivalence of labelled batching and the legacy scalar path."""

import pytest
import torch
from fiatlux import (
    Field,
    FieldDimension,
    Grid,
    Spectrum,
    FFTPropagator,
    MFTPropagator,
    SerialSystem,
    Detector,
    DetectorImage,
    ExposureAccumulator,
    Atmosphere,
    KolmogorovAtmosphereModel,
    ZeldaMask,
    ZeldaStop,
    ShackHartmannLensletArray,
    FatmossAtmosphereModel,
)
from fiatlux.core.spectrum import Band


def assert_roundoff_close(actual, expected):
    # Batched BLAS changes summation order. Near cancellations need an absolute
    # roundoff bound as well as relative tolerance, especially OPD derivatives
    # with units 1/metre. No output is renormalized.
    eps = torch.finfo(actual.real.dtype).eps
    torch.testing.assert_close(
        actual,
        expected,
        rtol=8 * eps,
        atol=8 * eps * float(expected.detach().abs().max()),
    )


def make_field(dtype=torch.float64, device="cpu", wavelengths=1):
    grid = Grid(8, 6, 0.1, 0.12, dtype=dtype, device=device)
    spectrum = Spectrum(0, Band(1e-6, 0.2e-6, 1), wavelengths).to(
        device=device, dtype=dtype
    )
    torch.manual_seed(18)
    a = torch.randn((wavelengths, 6, 8), device=device, dtype=dtype)
    return Field(torch.complex(a, a.sin()), grid, spectrum)


def time_dimension(n=4, dtype=torch.float64, device="cpu", dt=0.01):
    return FieldDimension(
        "time",
        n,
        torch.arange(n, dtype=dtype, device=device) * dt,
        "s",
        torch.full((n,), dt, dtype=dtype, device=device),
    )


@pytest.mark.parametrize("name", ["", " ", "wavelength", "x", "y"])
def test_invalid_names(name):
    with pytest.raises(ValueError):
        FieldDimension(name, 2)


def test_metadata_and_field_validation():
    f = make_field()
    t = time_dimension()
    with pytest.raises(ValueError, match="unique"):
        f.expand_dimension(t).expand_dimension(t)
    with pytest.raises(ValueError):
        FieldDimension("time", 2, torch.zeros(3))
    with pytest.raises(ValueError):
        FieldDimension("time", 2, integration_weights=torch.tensor([1.0, -1.0]))
    with pytest.raises(ValueError):
        Field(torch.zeros(2, 1, 6, 8), f.grid, f.spectrum)
    with pytest.raises(ValueError, match="dtype"):
        f.expand_dimension(time_dimension(dtype=torch.float32))
    b = f.expand_dimension(t)
    with pytest.raises(ValueError):
        b + b.rename_dimension("time", "mc")
    with pytest.raises(ValueError):
        f * torch.ones(4, 1, 6, 8)
    with pytest.raises(ValueError, match="Select"):
        b.plot()
    with pytest.raises(ValueError):
        f.apply_opd(torch.zeros(4, 6, 8, dtype=torch.float64))


def test_operations_and_named_opd_alignment():
    f = make_field()
    t, mc = time_dimension(), FieldDimension("mc", 2)
    b = f.expand_dimension(mc).expand_dimension(t)
    assert b.axis("time") == 1
    assert b.select("mc", -1).dimensions == (t,)
    assert b.slice("time", 1, 3).dimension("time").size == 2
    torch.testing.assert_close(
        b.slice("time", 1, 3).dimension("time").coordinates, t.coordinates[1:3]
    )
    torch.testing.assert_close(
        b.coherent_mean("time").select("mc", 0).complex_amplitude, f.complex_amplitude
    )
    torch.testing.assert_close(
        b.coherent_sum("time").select("mc", 0).complex_amplitude,
        4 * f.complex_amplitude,
    )
    moved = b.to(dtype=torch.complex64)
    assert moved.dimension("time").coordinates.dtype == torch.float32
    assert moved.dimension("time").integration_weights.dtype == torch.float32
    opd = torch.randn(4, 2, 6, 8, dtype=torch.float64) * 1e-8
    out = b.apply_opd(opd, dimensions=(t, mc))
    assert out.dimensions == (mc, t)
    torch.testing.assert_close(
        out.select("mc", 1).select("time", 2).complex_amplitude,
        f.apply_opd(opd[2, 1]).complex_amplitude,
    )
    torch.testing.assert_close((2 * b - b).complex_amplitude, b.complex_amplitude)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("method,wavelengths", [("fft", 1), ("mft", 3)])
@pytest.mark.parametrize(
    "regime,scale",
    [
        ("fraunhofer", 2.0),
        ("fraunhofer", -2.0),
        ("fresnel", 2.0),
        ("fresnel", -2.0),
        ("fresnel", 0.0),
    ],
)
def test_propagation_matches_loop_and_gradients(
    device, dtype, method, wavelengths, regime, scale
):
    f = make_field(dtype, device, wavelengths)
    t, mc = time_dimension(dtype=dtype, device=device), FieldDimension("mc", 2)
    opd = (torch.randn(2, 4, 6, 8, dtype=dtype, device=device) * 1e-8).requires_grad_()
    kwargs = dict(propagation=regime)
    (
        kwargs.update(focal_length=scale)
        if regime == "fraunhofer"
        else kwargs.update(distance=scale)
    )
    p = (
        FFTPropagator(**kwargs)
        if method == "fft"
        else MFTPropagator(
            output_grid=(
                f.grid
                if scale == 0
                else Grid(7, 5, 1e-6, 1.2e-6, device=device, dtype=dtype)
            ),
            **kwargs,
        )
    )
    batched = p.apply(f.apply_opd(opd, dimensions=(mc, t)))
    loop = torch.stack(
        [
            torch.stack(
                [p.apply(f.apply_opd(opd[i, j])).complex_amplitude for j in range(4)]
            )
            for i in range(2)
        ]
    )
    assert batched.dimensions == (mc, t)
    assert_roundoff_close(batched.complex_amplitude, loop)
    # A linear observable avoids a nearly-zero total-power gradient for FFT.
    g1 = torch.autograd.grad(
        batched.complex_amplitude.real.sum(), opd, retain_graph=True
    )[0]
    g2 = torch.autograd.grad(loop.real.sum(), opd)[0]
    assert_roundoff_close(g1, g2)


def test_incoherent_exposure_chunking_and_remaining_metadata():
    f = make_field(wavelengths=3)
    t, mc = time_dimension(), FieldDimension("mc", 2)
    b = f.expand_dimension(mc).expand_dimension(t)
    b = b * torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float64).reshape(
        1, 4, 1, 1, 1
    )
    detector = Detector(f.grid, exposure_time=0.04, quantum_efficiency=0.7)
    out = detector.acquire(b, integrate_over="time")
    assert isinstance(out, DetectorImage) and out.dimensions == (mc,)
    expected = f.intensity().sum(-3) * 0.04 * f.grid.dx * f.grid.dy * 0.7
    torch.testing.assert_close(out.data[0], expected)
    assert b.coherent_mean("time").intensity().sum() == 0
    accumulator = ExposureAccumulator(detector)
    accumulator.add(b.slice("time", 0, 1)).add(b.slice("time", 1, 4))
    torch.testing.assert_close(accumulator.finish().data, out.data)
    with pytest.raises(RuntimeError):
        accumulator.finish()
    with pytest.raises(ValueError, match="weights"):
        detector.acquire(
            f.expand_dimension(FieldDimension("time", 4)), integrate_over="time"
        )
    with pytest.raises(ValueError, match="exposure_time"):
        detector.acquire(b.slice("time", 0, 1), integrate_over="time")
    # Legacy image stays a tensor; unresolved samples remain labelled.
    assert isinstance(detector.acquire(f), torch.Tensor)
    assert isinstance(detector.acquire(b), DetectorImage)


def test_chunk_gradient_and_noise_once():
    f = make_field()
    amplitude = f.complex_amplitude.detach().requires_grad_()
    b = Field(amplitude, f.grid, f.spectrum).expand_dimension(time_dimension())
    d = Detector(f.grid, exposure_time=0.04)
    a = ExposureAccumulator(d, track_grad=True)
    a.add(b.slice("time", 0, 2)).add(b.slice("time", 2, 4))
    g1 = torch.autograd.grad(a.finish().sum(), amplitude, retain_graph=True)[0]
    g2 = torch.autograd.grad(d.acquire(b, integrate_over="time").sum(), amplitude)[0]
    torch.testing.assert_close(g1, g2)
    a = ExposureAccumulator(d).add(b)
    assert not a.finish().requires_grad
    kwargs = dict(
        exposure_time=0.04,
        photon_noise=True,
        dark_current=10,
        readout_noise_variance=3,
        digitize=True,
        random_seed=9,
    )
    d1, d2 = Detector(f.grid, **kwargs), Detector(f.grid, **kwargs)
    result = d1.acquire(b, integrate_over="time")
    a = ExposureAccumulator(d2).add(b.slice("time", 0, 2)).add(b.slice("time", 2, 4))
    torch.testing.assert_close(a.finish(), result, rtol=0, atol=0)


def test_atmosphere_buffer_seed_piston_and_system():
    f = make_field()
    model = KolmogorovAtmosphereModel(f.grid, 0.2, seed=31)
    atmosphere = Atmosphere(f.grid, model)
    state = model.generator.get_state()
    b = atmosphere.apply_buffer(f, 4, sample_period=0.01, advance=False)
    assert torch.equal(state, model.generator.get_state())
    opd = model.sample_opd_many(4)
    torch.testing.assert_close(
        opd.mean((-2, -1)), torch.zeros(4, dtype=torch.float64), atol=1e-20, rtol=0
    )
    torch.testing.assert_close(
        b.complex_amplitude, f.apply_opd(opd, dimensions=b.dimensions).complex_amplitude
    )
    p = FFTPropagator(2.0)
    result = SerialSystem([p]).run_field(b)
    assert result.final_field.dimensions == b.dimensions
    assert SerialSystem([]).run_field(f).final_field is f


def test_zelda_chain_and_dm_with_fixed_atmosphere_cube():
    from test_deformable_mirror_commands import make_dm

    dm = make_dm()
    grid = dm.grid
    spectrum = Spectrum(0, Band(1e-6, 0.2e-6, 1), 2)
    f = Field(torch.ones(2, 2, 3, dtype=torch.complex64), grid, spectrum)
    opd = torch.randn(4, 2, 3) * 1e-8
    t = time_dimension(dtype=torch.float32)
    b = f.apply_opd(opd, dimensions=(t,))
    p = MFTPropagator(2.0, grid)
    system = SerialSystem([dm, p])
    out = system.run_field(b).final_field
    expected = torch.stack(
        [
            system.run_field(f.apply_opd(screen)).final_field.complex_amplitude
            for screen in opd
        ]
    )
    torch.testing.assert_close(out.complex_amplitude, expected)
    g1 = torch.autograd.grad(
        out.complex_amplitude.real.sum(), dm._commands, retain_graph=True
    )[0]
    g2 = torch.autograd.grad(expected.real.sum(), dm._commands)[0]
    torch.testing.assert_close(g1, g2)
    # Actual ZELDA focal mask + return propagation + pupil stop.
    f = make_field()
    focal_grid = Grid(8, 6, 1e-6, 1e-6, dtype=torch.float64)
    chain = SerialSystem(
        [
            MFTPropagator(2.0, focal_grid),
            ZeldaMask(focal_grid, radius=1e-6, well_depth=2.5e-7),
            MFTPropagator(-2.0, f.grid),
            ZeldaStop(f.grid, radius=0.2),
        ]
    )
    opd = torch.randn(4, 6, 8, dtype=torch.float64) * 1e-8
    b = chain.run_field(f.apply_opd(opd, dimensions=(time_dimension(),))).final_field
    expected = torch.stack(
        [chain.run_field(f.apply_opd(v)).final_field.complex_amplitude for v in opd]
    )
    torch.testing.assert_close(b.complex_amplitude, expected)


def test_shack_hartmann_explicit_exclusion():
    f = make_field()
    sensor = ShackHartmannLensletArray(f.grid, pitch=0.6, focal_length=2.0)
    with pytest.raises(ValueError, match="latent"):
        sensor.propagate(f.expand_dimension(time_dimension()))


def test_fatmoss_correlated_buffer_and_partitions():
    from test_fatmoss import TranslatingBackend
    import types

    grid = Grid(8, 8, 0.1, 0.1, dtype=torch.float64)
    backend = TranslatingBackend(D=0.8, dx=0.1, dt=0.01)
    backend.layers = [types.SimpleNamespace(wind_speed=3.7)]
    model = FatmossAtmosphereModel(grid, backend, time_step=0.01)
    spectrum = Spectrum(0, Band(1e-6, 0, 1), 1).to(dtype=torch.float64)
    f = Field(torch.ones(1, 8, 8, dtype=torch.complex128), grid, spectrum)
    atmosphere = Atmosphere(grid, model)
    b = atmosphere.apply_buffer(f, 4, advance=False)
    assert model.current_time == 0
    p = FFTPropagator(2.0)
    propagated = p.apply(b)
    cube = model.sequence_opd(4, advance=False)
    loop = torch.stack([p.apply(f.apply_opd(v)).complex_amplitude for v in cube])
    torch.testing.assert_close(propagated.complex_amplitude, loop)
    d = Detector(propagated.grid, exposure_time=0.04)
    expected = d.acquire(propagated, integrate_over="time")
    for partition in ([1, 3], [2, 2]):
        model.seek(0)
        a = ExposureAccumulator(d)
        for n in partition:
            a.add(p.apply(atmosphere.apply_buffer(f, n)))
        torch.testing.assert_close(a.finish(), expected)
        assert model.current_time == 0.04


def test_accumulator_rejects_invalid_lifecycle_and_changed_metadata():
    f = make_field()
    b = f.expand_dimension(time_dimension()).expand_dimension(FieldDimension("mc", 2))
    d = Detector(f.grid, exposure_time=0.04)
    with pytest.raises(ValueError, match="empty"):
        ExposureAccumulator(d).finish()
    a = ExposureAccumulator(d).add(b.slice("time", 0, 1))
    with pytest.raises(ValueError, match="exposure_time"):
        a.finish()
    with pytest.raises(ValueError, match="dimensions"):
        a.add(b.slice("time", 1, 4).rename_dimension("mc", "other"))
    with pytest.raises(ValueError, match="exceed"):
        a.add(b)
    d.quantum_efficiency = 0.5
    with pytest.raises(ValueError, match="settings"):
        a.finish()


def test_masks_preserve_unrelated_axes_and_real_dtype_transfer_preserves_phase():
    from fiatlux import CircularAperture
    from fiatlux.optics.elements.mask import Piston

    f = make_field(wavelengths=3)
    b = f.expand_dimension(FieldDimension("mc", 2)).expand_dimension(time_dimension())
    for mask in (CircularAperture(f.grid, 0.3), Piston(f.grid, 1e-7)):
        out = mask.apply(b)
        assert out.dimensions == b.dimensions
        torch.testing.assert_close(
            out.select("mc", 1).select("time", 3).complex_amplitude,
            mask.apply(f).complex_amplitude,
        )
    torch.testing.assert_close(
        b.to(dtype=torch.float32).complex_amplitude,
        b.complex_amplitude.to(torch.complex64),
    )


def test_opd_gradcheck_and_metadata_replay():
    f = make_field()
    t = time_dimension(2)
    opd = torch.zeros(2, 6, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda v: f.apply_opd(v, dimensions=(t,)).complex_amplitude,
        (opd,),
        eps=1e-10,
        atol=1e-3,
        rtol=1e-5,
    )
    m1 = KolmogorovAtmosphereModel(f.grid, 0.2, seed=12)
    m2 = KolmogorovAtmosphereModel(f.grid, 0.2, seed=12)
    torch.testing.assert_close(
        m1.sample_opd_many(3), m2.sample_opd_many(3), rtol=0, atol=0
    )
