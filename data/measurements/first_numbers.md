# First PCMO numbers (small blob classifier, two 3-bit linear layers)

Quantized (QAT, no memristors) test accuracy: **0.979**. 5 programming seeds per row; mean +- std over seeds.

| backend | encoding | readout noise | accuracy | drop vs QAT |
|---|---|---|---|---|
| Synaptogen | amplitude 0.6 V | legacy | 0.977 +- 0.001 | +0.002 |
| Synaptogen | amplitude 0.6 V | off | 0.977 +- 0.002 | +0.002 |
| W/WOx/PCMO | pulse count at -0.2 V (LRS / pristine) | off | 0.973 +- 0.001 | +0.006 |
| W/WOx/PCMO | pulse count at -0.2 V (LRS / pristine) | physical, BW 1e8 Hz | 0.972 +- 0.004 | +0.007 |
| W/WOx/PCMO | pulse count at -0.2 V (LRS / pristine) | physical, BW 1e6 Hz | 0.973 +- 0.001 | +0.006 |
| Ta/WOx/PCMO | pulse count at -0.2 V (LRS / pristine) | off | 0.976 +- 0.002 | +0.003 |
| Ta/WOx/PCMO | pulse count at -0.2 V (LRS / pristine) | physical, BW 1e8 Hz | 0.977 +- 0.001 | +0.002 |
| Ta/WOx/PCMO | pulse count at -0.2 V (LRS / pristine) | physical, BW 1e6 Hz | 0.977 +- 0.001 | +0.002 |

## Device pools

- **W/WOx/PCMO**: 76 devices, 1 cycle each; median I_on 1.15e-10 A, I_off 7.93e-12 A at -0.2 V (on/off 14.5); device-to-device spread of I_on 0.11 dex; unit-weight conductance 5.18e-10 S.
- **Ta/WOx/PCMO**: 12 devices, 1 cycle each; median I_on 3.43e-10 A, I_off 2.5e-11 A at -0.2 V (on/off 13.7); device-to-device spread of I_on 0.06 dex; unit-weight conductance 1.61e-09 S.

Disclaimers: first-loop data only, pristine branch as the off state, no cycle-to-cycle data, 76 / 12 devices reused across all cells, no retention or read disturb, instrument floor not modelled.
