# Datasets

All files are plain comma-separated CSV with a dot decimal separator. Time is
in hours since the start of the experiment (`Time (h)`), sampled every 30 s
(0.008333 h). Temperatures are in °C.

## Sensor naming

Soil thermistor columns follow `H<depth_mm>D<radial_mm>`. For example,
`H90D45` is the thermistor 90 mm below the surface and 45 mm from the heat
exchanger pipe axis. The replay tool turns this into the twin's coordinates
`z_m = depth / 1000` and `x_m = radial / 1000`, so the column header is
the only sensor map the replay path needs.

| Folder | Files | Replayable | Notes |
|---|---|---|---|
| `april8_heating/` | `sensors.csv` (74 thermistors), `fluid.csv` (`Inlet`, `Outlet`) | **yes**: reference demo | 6.6 h heating run used to calibrate and validate the twin. The circulator inlet/outlet temperatures drive the pipe boundary model. |
| `heat_exchanger_test/` | `sensors.csv` (75 thermistors), `fluid.csv`, `logger_reference_temp.csv`, `soil_vwc.csv` | **yes** | 23 h heat-exchanger test. `soil_vwc.csv` holds volumetric water content, EC and probe temperature at depths of 70, 130 and 210 mm. `logger_reference_temp.csv` holds the multiplexer reference temperatures. |
| `frozen_saturated/` | `temperature.csv`, `soil_vwc.csv`, `heat_flux.csv` | no | Earlier frozen, saturated sample test. Its columns use the older `D<depth>-W<width>` naming and a different sensor layout. It is provided as reference data only and is not wired into the replay tool. |

> **Note on `H10D75`.** This thermistor failed during the campaign. It is
> disabled for live acquisition in `startup.conf`, but it is still present in
> the replay CSVs. To leave it out of a replay, delete that column from a copy
> of `sensors.csv`.

## Bringing your own data

- **Logged CR1000 table** (`TIMESTAMP` column, `Inlet`/`Outlet` inline):
  ```bash
  python -m software.tools.prep_cr1000_replay --path my_run.csv --out-dir data/local
  ```
  This writes `<name>_replay.csv` and `<name>_fluid.xlsx`, which you can pass
  to `replay_csv`.
- **Any other source:** produce a CSV with a numeric `Time (h)` column, one
  `H<depth>D<radial>` column per sensor and, optionally, a second file with
  `Time (h)`, `Inlet` and `Outlet`.

Put private or unpublished data in `data/local/`. Git ignores that folder.
