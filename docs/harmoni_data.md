# HARMONI residual dataset format

`HarmoniResiduals` loads a directory of FITS files in deterministic filename
order. The directory is supplied explicitly:

```python
residuals = HarmoniResiduals(
    grid=pupil_grid,
    dataset_path="/path/to/harmoni_residuals",
)
```

## Default FITS structure

By default, each primary HDU contains a floating-point array shaped
`(n_products, n_screens, ny, nx)`. Product index 1 contains residual optical
path difference. Values are interpreted as metres of OPD. Files, HDUs, and
arrays that do not follow the declared structure are rejected with a
`HarmoniDatasetError` subclass naming the offending path.

The defaults reproduce the historical orientation by rotating each screen by
one quarter turn. Set `rotate_quarter_turns=0` if the stored orientation already
matches the Fiatlux grid. The transformed spatial shape must equal
`(grid.ny, grid.nx)`.

For FITS files storing a direct cube shaped `(n_screens, ny, nx)`, pass
`opd_plane_index=None`. A dataset requiring unit conversion can use an explicit
factor such as `opd_scale=1e-9` for nanometres to metres. The default scale is
1, so no undocumented physical conversion is applied.

## Pupil support and screen order

Pass an authoritative boolean or binary support as `pupil=` whenever the
dataset supplies one. Otherwise Fiatlux derives a compatibility support from
the union of non-zero pixels over the complete loaded cube.

Screens are visited without replacement in a seeded permutation and then
cycled. Use `seed=` for repeatable shuffled order or `shuffle=False` for strict
file-and-frame order.

Install FITS support with:

```bash
python -m pip install 'fiatlux[fits]'
```
