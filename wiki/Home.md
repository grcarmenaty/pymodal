# pymodal

**Build large, labelled collections of vibrational signals on disk and feed them straight into PyTorch — without loading the dataset into RAM.**

`pymodal` is a Python library developed at the Applied Mechanics Lab, IQS School of Engineering, for Structural Health Monitoring (SHM) research. Its primary purpose is to assemble, persist, pre-process, and label collections of time-domain signals and Frequency Response Functions (FRFs), and to expose those collections as `torch.utils.data.Dataset` objects ready for training.

## The intended workflow

1. **Acquire or simulate** vibrational measurements (time-series or FRFs).
2. **Assemble** them into a `timeseries` or `frf` collection — a single `ndarray` builds a 1-item collection; a list of `ndarray`s builds a multi-item collection. Every measurement is written to an HDF5 file on disk as it is added.
3. **Pre-process** in place: resample, change span, augment with Gaussian noise, convert time-series to FRF. All operations stream through HDF5 — the full dataset never has to live in memory.
4. **Compute SHM indicators** (CFDAC, RVAC, SCI, …). Each call returns a typed indicator collection (`cfdac_collection`, `sci_collection`, …) with the source `reference` and `damaged` collections embedded for full provenance.
5. **Hand off to PyTorch** with `.torch_dataset()` to obtain an `HDF5Dataset` that lazy-loads samples on demand.

```python
import numpy as np
import pymodal

resp = pymodal.timeseries([acc1, acc2], time_step=1/8192)
exc  = pymodal.timeseries([f1, f2],     time_step=1/8192, method="excitation")

frfs = resp.to_FRF(exc, FRF_type="H1")          # frf collection
sci  = frfs[1:].sci(reference=frfs[:1])         # 0-D indicator collection

dataset = sci.torch_dataset().dataset           # ready for DataLoader
```

## Where to start

- **New here?** Read [Core Concepts](Core-Concepts) first — the dimensional design and HDF5 layout decide everything else.
- **Want to run code now?** Jump to [Installation](Installation) and then [Quickstart](Quickstart).
- **Looking for a specific class or function?** See the [API Reference](API-Reference).

## Wiki navigation

| Page | What it covers |
|---|---|
| [Installation](Installation) | `pip install`, dev install, optional MCP extra, dependencies. |
| [Core Concepts](Core-Concepts) | Dimensional collections, item shape, HDF5 layout, references, units. |
| [Quickstart](Quickstart) | End-to-end example: build → augment → compute FRFs → indicator → PyTorch. |
| [Collections](Collections) | The `_collection_0d/1d/2d/3d` parents, names/labels, `append`, `split`, indexing, file lifecycle. |
| [Timeseries](Timeseries) | Time-domain collections, sampling rate, span, `AddGaussianNoise`, `to_FRF`. |
| [FRF](FRF) | Frequency-domain collections, plotting, waterfall, indicator methods. |
| [Indicators](Indicators) | All 0-D / 1-D / 2-D SHM indicators, formulas, references. |
| [HDF5 Dataset](HDF5-Dataset) | The PyTorch wrapper: lazy loading, RAM cache, `pad_collate_fn`. |
| [MCP Server](MCP-Server) | Driving pymodal from an LLM agent over Model Context Protocol. |
| [API Reference](API-Reference) | Index of public symbols by module. |
| [FAQ](FAQ) | Common questions and pitfalls. |

## Project status

pymodal is research code maintained on the `master` branch. The library is pre-1.0 and the API can change between releases; pin a specific version when reproducibility matters.

## License

MIT. See `LICENSE` in the repository.
