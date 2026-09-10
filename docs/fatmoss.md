# Optional FATMOSS integration

Fiatlux integrates [FATMOSS](https://github.com/EjjeSynho/FATMOSS) through a
narrow optional adapter. Importing and using core Fiatlux does not import or
require FATMOSS.

The adapter currently targets the upstream `main` API at commit
`12e608686dfc69955f722e2ee62455fcd3e32d2e`, in particular
`PhaseScreensGenerator(...)` and `GetScreenByTimestep(timestep)`.

## Installation

FATMOSS does not currently publish an installable Python package. Clone it and
make its source directory importable:

```bash
git clone https://github.com/EjjeSynho/FATMOSS.git
export PYTHONPATH="$PWD/FATMOSS:$PYTHONPATH"
```

The upstream `settings.json` controls NumPy/CuPy execution and must remain next
to the FATMOSS source files. Install its dependencies according to its README.

## Constructing the adapter

```python
from fiatlux import FatmossAtmosphereModel, Grid

grid = Grid(256, 256, 0.04, 0.04)
model = FatmossAtmosphereModel.create(
    grid,
    time_step=1e-3,
    batch_size=100,
    n_cascades=3,
    seed=42,
)
```

## Frozen-flow sequences

Use a Fiatlux-owned physical configuration instead of constructing FATMOSS
layers directly:

```python
from fiatlux import FatmossAtmosphereModel, FrozenFlowLayer

model = FatmossAtmosphereModel.create_frozen_flow(
    grid,
    [FrozenFlowLayer(
        r0=0.15,
        outer_scale=25.0,
        wind_speed=10.0,
        wind_direction=45.0,
    )],
    time_step=1e-3,
    seed=42,
)

opd_now = model.current_opd()       # metres; does not advance time
phase_now = model.current_phase()   # radians at reference_wavelength
model.advance()
opd_at_10_ms = model.opd_at(0.010)  # state is unchanged by the query
cube = model.sequence_opd(100)      # (time, ny, nx), then advances 100 steps
```

`wind_speed` is in m/s and `wind_direction` is in degrees, following FATMOSS.
The physical displacement per cadence is passed to FATMOSS without rounding,
so fractional-pixel frozen-flow translations are preserved. A time supplied to
`opd_at()`, `phase_at()` or `seek_time()` must lie exactly on the configured
cadence.

Optical evaluation and time advancement are deliberately separate for models
created by `create_frozen_flow()`: repeated calls to `sample_opd()` return the
same instant until `advance()` or `seek_time()` is called. This makes it possible
to propagate multiple wavelengths or optical branches through one atmosphere
state without accidentally moving the turbulence between evaluations.

## Conventions

- `Grid.dx`, `Grid.dy` and generator diameter `D`: metres.
- `time_step`: seconds.
- `FrozenFlowLayer.r0`, `outer_scale`, `altitude`: metres.
- `FrozenFlowLayer.wind_speed`: m/s; `wind_direction`: degrees.
- `reference_wavelength`: metres in Fiatlux; FATMOSS currently uses 500 nm.
- upstream screens: nanometres of OPD in `(x, y)` order.
- `sample_opd()` output: metres of OPD in Fiatlux `(ny, nx)` order.
- the FATMOSS seed owns temporal reproducibility; a PyTorch generator is not
  accepted by `sample_opd()`.

No rescaling to a target RMS or modification of the FATMOSS PSD is performed.
