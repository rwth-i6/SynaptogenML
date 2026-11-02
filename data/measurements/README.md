# Partner IV measurements (received 2026-11-02)

Ta/WOx/PCMO and W/WOx/PCMO crossbar devices, integrated as 1T1R cells on the
Neurotec chip. Each file contains **the first IV loop of every device**
(pristine devices, one loop each). Analysis: `python tools/analyze_iv_dat.py
data/measurements/*.dat --plots data/measurements/plots`.

| File | Devices | Rows/device | Loop | Date | Hold per point |
|---|---|---|---|---|---|
| `Ta_WOx_PCMO.dat` | 12 | 191-201 | 0 → -1.5 → +1.35..1.5 → 0 V, 30 mV steps | 2026-04-13 | 30 ms |
| `W_WOx_PCMO.dat` | 76 | 201 | 0 → -1.5 → +1.5 → 0 V, 30 mV steps | 2026-02-26 | 20 ms |

Format: one ASCII header, then one tab-separated table with columns
`Voltage Source (V)`, `Voltage Probe (V)`, `Current Probe (A)`, `Resistance (Ohm)`.
All devices are concatenated; a device starts at the `0.0` sample followed by a
negative sample. Source and probe voltage differ by < 1.2 mV, use the source
column. Compliance (100 mA) is never reached. Current auto-ranging ("Spikes!")
leaves visible range-switch glitches near 0 V.

## Device behaviour (both stacks)

- **Bipolar, gradual (interface-type) switching**: negative voltage SETs
  (conductance rises continuously beyond about -0.5 V for Ta and -0.85 V for W),
  positive voltage RESETs around +0.5 V (Ta) / +1.0 V (W). No abrupt jumps.
- **Strongly rectifying**: at +1.5 V the current is only nA, at -1.5 V it is µA.
  In the LRS the current at -0.2 V is 7x (W) to 45x (Ta) larger than at +0.2 V.
  Positive-polarity reads are at or near the instrument floor.
- **Very high resistance**: median LRS at -0.2 V is 0.58 GΩ (Ta) / 1.7 GΩ (W),
  pristine state 8 GΩ (Ta) / 25 GΩ (W). Read currents in a ±0.6 V window are
  10 pA to a few nA. Instrument noise floor is about 4-5 pA.
- **Nonlinear IV**: LRS current grows 10x (W) / 48x (Ta) from -0.2 V to -0.5 V,
  where an ohmic device would give 2.5x.
- **On/off at -0.2 V** (LRS vs pristine): about 14x in both stacks.
- **Device-to-device spread**: std of log10 LRS current at -0.2 V is 0.11 dex
  (W) / 0.06 dex (Ta). The first 9 devices of the W file form a distinct
  subpopulation (3x lower LRS current, 2.5x higher pristine current, SET onset
  at -0.77 V instead of -0.89 V).

## What is missing for the device model

- **No cycle-to-cycle data**: one loop per device.
- **No HRS read at negative polarity**: the loop ends at 0 V after the positive
  RESET half, so the post-reset state is only seen at positive polarity, where
  the current is at the noise floor. The pristine branch is the only negative-
  polarity "high resistance" read and the first cycle is usually not
  representative (forming).
- **No intermediate states**, no retention/read-disturb information, no pulse
  programming data.

Requested from the partner for the next delivery: at least 2-3 consecutive
loops per device (so HRS after RESET is read at negative polarity), ideally
tens of loops on a subset of devices, and pulse-programmed read sweeps at the
intended read voltage.
