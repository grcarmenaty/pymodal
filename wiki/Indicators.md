# SHM Indicators

pymodal ships sixteen damage indicators drawn from the SHM literature, organised into three families by dimensionality. Each indicator is implemented twice:

1. **As a pure function** in `pymodal.utils` operating on `(n_dof, n_freq)` complex matrices. These are the single source of truth for the math.
2. **As a typed indicator collection** that pairs a `reference` and `damaged` `frf` collection, applies the pure function per item, and stores the result with channel/DOF metadata recovered from the embedded references.

You normally use the **method form on `frf`** — never instantiate the collection class directly:

```python
sci  = damaged_frfs.sci(reference=baseline_frfs)
cfdac = damaged_frfs.cfdac(reference=baseline_frfs)
```

The result is an HDF5-backed collection with the same `name`, `labels`, channel metadata, and `(reference, damaged)` references embedded.

## How a (reference, damaged) pair becomes an indicator

```python
# Conceptually, for every damaged item i:
ref_H = reference.measurements[ref_idx[i]][()]     # (n_freq, n_outputs, n_inputs)
dmg_H = damaged.measurements[i][()]
result = pure_function(_as_matrix(ref_H), _as_matrix(dmg_H))
# result.shape ∈ {(),  (n_dof,),  (n_dof, n_dof)}
# stored as (1, 1) / (n_dof, 1, 1) / (n_dof, n_dof, 1, 1) respectively
```

`_as_matrix(arr)` reshapes a `(n_freq, n_outputs, n_inputs)` FRF into `(n_dof, n_freq)` via `arr.reshape(n_freq, -1).T`.

`ref_idx` is `[0, 1, 2, …]` when the reference and damaged collections have the same length, or `[0, 0, 0, …]` when the reference is a single baseline against which every damaged item is compared.

## 0-D indicators — one scalar per item

Item shape on disk: `(1, 1)`. Channel/DOF metadata is preserved via the embedded references, not duplicated on the indicator values.

| Method on `frf` | Class | Pure function | What it measures |
|---|---|---|---|
| `sci(ref)` | `sci_collection` | `utils.SCI(\|CFDAC(ref,ref)\|, \|CFDAC(ref,dmg)\|)` | Signed structural-change indicator (`k · (1 − \|PCC\|)`) |
| `unsigned_sci(ref)` | `unsigned_sci_collection` | `utils.unsigned_SCI(...)` | Unsigned variant: `1 − \|PCC\|` |
| `drq(ref)` | `drq_collection` | `utils.DRQ(RVAC)` | Mean of the RVAC vector |
| `aigac(ref)` | `aigac_collection` | `utils.AIGAC(GAC)` | Mean of the GAC vector |
| `frfrms(ref)` | `frfrms_collection` | `utils.FRFRMS(...)` | RMS of `log10`-magnitude differences |
| `frfsf(ref)` | `frfsf_collection` | `utils.FRFSF(...)` | Ratio of summed magnitudes |
| `frfsm(ref, std=6.0)` | `frfsm_collection` | `utils.FRFSM(..., std=...)` | Gaussian similarity in dB (parameter is std in dB) |
| `ods_diff(ref)` | `ods_diff_collection` | `utils.ODS_diff(...)` | Summed absolute ODS difference |
| `r2_imag(ref)` | `r2_imag_collection` | `utils.r2_imag(...)` | R² of the imaginary part |

`frfsm` is the only 0-D indicator that takes a parameter beyond the reference: `std` (default `6.0`, in dB) controls the Gaussian width.

## 1-D indicators — one vector per item, indexed over DOF

Item shape on disk: `(n_dof, 1, 1)`. The single domain axis represents the DOF index inherited from the source FRFs (units `"dof_index"`).

| Method on `frf` | Class | Pure function | What it measures |
|---|---|---|---|
| `rvac(ref)` | `rvac_collection` | `utils.value_RVAC(ref, dmg)` | Per-DOF Response Vector Assurance Criterion |
| `rvac_2d(ref)` | `rvac_2d_collection` | `utils.value_RVAC_2d(...)` | Curvature variant — second-difference RVAC |
| `gac(ref)` | `gac_collection` | `utils.value_GAC(...)` | Per-DOF Global Amplitude Criterion |
| `m2l(ref)` | `m2l_collection` | `utils.M2L(\|CFDAC(ref,dmg)\|)` | Mode-shape-to-Local damage indicator |

## 2-D indicators — one matrix per item, indexed over DOF × DOF

Item shape on disk: `(n_dof, n_dof, 1, 1)`. Two domain axes named `"dof_index"`. CFDAC results are stored as `complex64`; FDAC as `float64`.

| Method on `frf` | Class | Pure function | What it measures |
|---|---|---|---|
| `cfdac(ref)` | `cfdac_collection` | `utils.value_CFDAC(ref, dmg)` | Complex Frequency Domain Assurance Criterion |
| `cfdac_a(ref)` | `cfdac_a_collection` | `utils.value_CFDAC_A(...)` | CFDAC alternative formulation |
| `fdac(ref)` | `fdac_collection` | `utils.value_FDAC(...)` | Real-valued FDAC |

## Pure-function reference

Every indicator delegates to a pure function on `(n_dof, n_freq)` matrices in `pymodal.utils`. They are exported from the top-level `pymodal` namespace, so you can call them directly on raw arrays without going through a collection.

```python
import numpy as np
import pymodal

# H_ref, H_dmg shaped (n_dof, n_freq)
cfdac_mat = pymodal.value_CFDAC(H_ref, H_dmg)            # (n_freq, n_freq) complex
rvac_vec  = pymodal.value_RVAC(H_ref, H_dmg)             # (n_dof,) float
gac_vec   = pymodal.value_GAC(H_ref, H_dmg)              # (n_dof,) float
fdac_mat  = pymodal.value_FDAC(H_ref, H_dmg)             # (n_freq, n_freq) real

drq    = pymodal.DRQ(rvac_vec)
aigac  = pymodal.AIGAC(gac_vec)
sci    = pymodal.SCI(np.abs(pymodal.value_CFDAC(H_ref, H_ref)),
                      np.abs(pymodal.value_CFDAC(H_ref, H_dmg)))
sci_u  = pymodal.unsigned_SCI(...)                       # same call signature

frfrms = pymodal.FRFRMS(H_ref, H_dmg)
frfsf  = pymodal.FRFSF(H_ref, H_dmg)
frfsm  = pymodal.FRFSM(H_ref, H_dmg, std=6.0)
ods    = pymodal.ODS_diff(H_ref, H_dmg)
r2     = pymodal.r2_imag(H_ref, H_dmg)

m2l_vec = pymodal.M2L(np.abs(cfdac_mat))                 # (n_dof,)
```

## References

The published formulations the indicators implement:

- **FRFRMS** — Sampaio et al., *Mechanical Systems and Signal Processing* (2002): https://www.sciencedirect.com/science/article/abs/pii/S1270963802011938
- **SCI** — Garcia-Macias et al. (2018): https://www.sciencedirect.com/science/article/abs/pii/S0888327018306551
- The remaining indicators (FDAC, RVAC, GAC, AIGAC, DRQ, FRFSF, FRFSM, ODS-diff, R²-imag, CFDAC, M2L) are standard in the FRF-based SHM literature; the implementations follow the formulas in `utils.py` directly.

## Picking an indicator

The choice of indicator depends on what you want to detect and how much information you keep per item.

- **Want a single number per realisation?** Use 0-D — typically `sci` for signed change, `drq` or `aigac` for mean correlation, `frfrms` for amplitude difference.
- **Want to localise the damage along a chain of DOFs?** Use 1-D — `rvac` or `gac` for direct per-DOF correlation, `m2l` for an indicator designed specifically for localisation.
- **Want full off-diagonal coupling structure (typically as input to a 2-D CNN)?** Use 2-D — `cfdac` is the standard choice; `fdac` is its real-valued counterpart.

Multiple indicators on the same pair are cheap — each call writes its own HDF5 file with the same embedded references, so you can compute a panel of complementary metrics and use them as multi-task labels or stacked features.

## Pitfalls

- **The reference and damaged collections must share the same frequency grid** for the matrix products to align. Crop or resample (`change_freq_span`, `change_freq_resolution`) before computing.
- **Indicators store channel shape `(1, 1)` on disk.** This is a deliberate choice — the channel/DOF metadata travels with the embedded `reference` collection, which preserves the original `n_outputs` × `n_inputs` and the spatial coordinates. Use `coll.references["reference"]` to recover them.
- **`PCC` (Pearson correlation) becomes ill-defined when one of the matrices is constant.** SCI and unsigned-SCI wrap the result in `np.nan_to_num`; you may still see warning suppressions during construction.
- **`m2l` operates on `np.abs(CFDAC)`** internally — the indicator itself does the absolute-value step before calling `utils.M2L`.
