"""FastMCP server exposing pymodal as a set of tools.

Design
------
Pymodal collections are HDF5-backed: every collection has a backing ``.h5``
file that is created on construction and that contains everything needed to
reconstruct the collection — items, channel/DOF metadata, domain axes, and
embedded reference snapshots. The MCP layer treats those files as the single
source of identity. Every tool takes file paths in and writes file paths out;
no in-memory collection objects survive between calls. This keeps the server
stateless and makes the tools idempotent and easy to compose into pipelines.

A typical pipeline looks like:

1. ``create_timeseries_collection`` — stack raw ``.npy`` arrays into one
   labelled ``timeseries`` HDF5.
2. ``add_gaussian_noise`` — augment the time-domain collection in place.
3. ``timeseries_to_frf`` — pair with an excitation collection and compute
   the FRFs (the resulting ``frf`` keeps both ``input`` and ``output`` as
   embedded references).
4. ``compute_indicator`` — produce a typed indicator collection (``cfdac``,
   ``rvac``, ``sci``, …) for a (reference, damaged) FRF pair.
5. ``split_collection`` — stratified train/val/test split stored on the
   collection.
6. The HDF5 file at the final path is then ready for ``HDF5Dataset`` and a
   PyTorch ``DataLoader`` (see :func:`torch_dataset_summary`).

Tool philosophy
---------------
- All paths are returned as ``str`` (absolute) so the agent can chain calls.
- Every tool that produces a new collection also returns a description of
  the resulting file (shape, names, labels, references, …) so the agent
  doesn't need a second ``describe_collection`` call to know what it just
  built.
- Numpy arrays are never serialised through MCP. Heavy data lives on disk;
  only metadata flows through the protocol.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from mcp.server.fastmcp import FastMCP

import pymodal
from pymodal.mcp.loaders import (
    describe_collection as _describe,
    load_frf,
    load_timeseries,
)


mcp = FastMCP("pymodal")


# ── helpers ───────────────────────────────────────────────────────────────────

def _abs(path: Optional[str]) -> Optional[str]:
    """Resolve ``path`` to an absolute string (or return ``None``)."""
    if path is None:
        return None
    return str(Path(path).expanduser().resolve())


def _load_arrays(data_paths: list[str]) -> list[np.ndarray]:
    """Load every path with :func:`pymodal.load_array` (handles npy/npz/mat)."""
    arrs: list[np.ndarray] = []
    for p in data_paths:
        a = pymodal.load_array(_abs(p))
        if isinstance(a, dict):
            # Handle .npz / .mat that return dicts: take the first non-meta entry.
            keys = [k for k in a.keys() if not str(k).startswith("__")]
            if not keys:
                raise ValueError(f"No usable arrays found in {p}")
            a = a[keys[0]]
        arrs.append(np.asarray(a))
    return arrs


def _close_keep(coll) -> str:
    """Close ``coll``'s file (keeping it on disk) and return the resolved path."""
    path = str(Path(coll.path).resolve())
    coll.close(keep=True)
    return path


# ── inspection ────────────────────────────────────────────────────────────────

@mcp.tool()
def describe_collection(path: str) -> dict:
    """Return a metadata summary of the pymodal HDF5 collection at ``path``.

    Reports n_items, item_shape, channel layout (method, n_outputs, n_inputs,
    dof), the domain axes (length / min / max / resolution / units), labels,
    reference roles, and any attached files (e.g. saved PyTorch state dicts).
    Works for any rank of collection (0-D scalar, 1-D frf/timeseries,
    2-D cfdac/fdac, …) without needing to know which class produced it.
    """
    return _describe(_abs(path))


@mcp.tool()
def list_collection_classes() -> dict:
    """Enumerate every concrete collection class pymodal exposes.

    Useful as a discovery tool for an agent that wants to know what kinds of
    artifacts it can produce (FRFs, time-series, the various indicator
    families) and the rank of items each class stores.
    """
    return {
        "raw": [
            {"class": "frf", "rank": 1, "domain": "frequency"},
            {"class": "timeseries", "rank": 1, "domain": "time"},
        ],
        "indicators_0d": [
            {"class": c, "rank": 0}
            for c in [
                "sci_collection", "unsigned_sci_collection", "drq_collection",
                "aigac_collection", "frfrms_collection", "frfsf_collection",
                "frfsm_collection", "ods_diff_collection", "r2_imag_collection",
            ]
        ],
        "indicators_1d": [
            {"class": c, "rank": 1}
            for c in [
                "rvac_collection", "rvac_2d_collection",
                "gac_collection", "m2l_collection",
            ]
        ],
        "indicators_2d": [
            {"class": c, "rank": 2}
            for c in ["cfdac_collection", "cfdac_a_collection", "fdac_collection"]
        ],
    }


@mcp.tool()
def list_indicators() -> dict:
    """List the SHM damage indicators available on :class:`pymodal.frf`.

    Returns the method name (callable as ``frf.<name>(reference)``) and a
    short description for each. Group keys (``"0d"``/``"1d"``/``"2d"``)
    describe the rank of the resulting indicator items.
    """
    return {
        "0d": {
            "sci": "Signed Correlation Index — scalar per (reference, damaged) pair.",
            "unsigned_sci": "Unsigned variant of SCI.",
            "drq": "Damage Relative Quotient.",
            "aigac": "Average Integrated Global Amplitude Criterion.",
            "frfrms": "FRF Root-Mean-Square deviation from reference.",
            "frfsf": "FRF Shape Factor.",
            "frfsm": "FRF Statistical Mean (takes a `std` parameter, default 6.0).",
            "ods_diff": "Operating Deflection Shape difference.",
            "r2_imag": "Coefficient of determination on imaginary parts.",
        },
        "1d": {
            "rvac": "Response Vector Assurance Criterion (1-D over DOF).",
            "rvac_2d": "Curvature variant of RVAC.",
            "gac": "Global Amplitude Criterion (1-D over DOF).",
            "m2l": "Mode-shape-to-Local damage indicator (1-D).",
        },
        "2d": {
            "cfdac": "Complex Frequency Domain Assurance Criterion (2-D matrix).",
            "cfdac_a": "Amplitude-only CFDAC variant.",
            "fdac": "Frequency Domain Assurance Criterion (real 2-D matrix).",
        },
    }


# ── construction ──────────────────────────────────────────────────────────────

@mcp.tool()
def create_frf_collection(
    data_paths: list[str],
    output_path: str,
    freq_start: float = 0.0,
    freq_end: Optional[float] = None,
    freq_resolution: Optional[float] = None,
    method: str = "SIMO",
    measurements_units: str = "millimeter / second ** 2 / newton",
    freq_units: str = "hertz",
    space_units: str = "millimeter",
    labels: Optional[list[float]] = None,
    names: Optional[list[str]] = None,
) -> dict:
    """Build a :class:`pymodal.frf` HDF5 collection from one or more arrays on disk.

    Each path in ``data_paths`` is loaded with :func:`pymodal.load_array`
    (``.npy`` / ``.npz`` / ``.mat`` supported) and contributes one item to the
    collection. Every array must share the same ``(n_freq, n_outputs,
    n_inputs)`` shape. Provide either ``freq_resolution`` or ``freq_end`` to
    define the frequency axis (``freq_start`` defaults to 0). ``labels`` is
    optional but enables stratified splitting and PyTorch ``(x, y)`` batches.
    """
    arrays = _load_arrays(data_paths)
    coll = pymodal.frf(
        measurements=arrays,
        freq_start=freq_start,
        freq_end=freq_end,
        freq_resolution=freq_resolution,
        method=method,
        measurements_units=measurements_units,
        freq_units=freq_units,
        space_units=space_units,
        labels=labels,
        names=names,
        path=_abs(output_path),
    )
    path = _close_keep(coll)
    return _describe(path)


@mcp.tool()
def create_timeseries_collection(
    data_paths: list[str],
    output_path: str,
    sampling_rate: Optional[float] = None,
    time_step: Optional[float] = None,
    time_start: float = 0.0,
    time_end: Optional[float] = None,
    method: str = "SIMO",
    measurements_units: Optional[str] = None,
    time_units: str = "second",
    space_units: str = "millimeter",
    labels: Optional[list[float]] = None,
    names: Optional[list[str]] = None,
) -> dict:
    """Build a :class:`pymodal.timeseries` HDF5 collection from arrays on disk.

    Provide either ``sampling_rate`` (Hz) or ``time_step`` (seconds) to set
    the time axis. Use ``method="excitation"`` to mark a force-input collection
    (its default ``measurements_units`` is ``"newton"``); response collections
    default to ``"meter / second ** 2"`` (acceleration).
    """
    arrays = _load_arrays(data_paths)
    if time_step is None and sampling_rate is not None:
        time_step = 1.0 / float(sampling_rate)
    coll = pymodal.timeseries(
        measurements=arrays,
        time_start=time_start,
        time_end=time_end,
        time_step=time_step,
        method=method,
        measurements_units=measurements_units,
        time_units=time_units,
        space_units=space_units,
        labels=labels,
        names=names,
        path=_abs(output_path),
    )
    path = _close_keep(coll)
    return _describe(path)


@mcp.tool()
def append_to_collection(
    input_path: str,
    output_path: str,
    data_path: str,
    name: Optional[str] = None,
    label: Optional[float] = None,
    kind: str = "frf",
) -> dict:
    """Append one new array as an extra item to an existing 1-D collection.

    Reads ``input_path``, reconstructs the collection (as ``frf`` or
    ``timeseries`` per ``kind``), appends the array loaded from ``data_path``
    with optional ``name`` and ``label``, and writes the result to
    ``output_path``. The source file is not modified.
    """
    if kind not in ("frf", "timeseries"):
        raise ValueError("kind must be 'frf' or 'timeseries'")
    loader = load_frf if kind == "frf" else load_timeseries
    coll = loader(_abs(input_path), output_path=_abs(output_path))
    new_arr = _load_arrays([data_path])[0]
    coll.append(new_arr, name=name, label=label)
    path = _close_keep(coll)
    return _describe(path)


# ── domain edition ────────────────────────────────────────────────────────────

@mcp.tool()
def change_freq_resolution(
    input_path: str,
    output_path: str,
    new_resolution: float,
) -> dict:
    """Resample every FRF in the collection to ``new_resolution`` (in freq_units).

    Equivalent to ``pymodal.frf.change_freq_resolution`` applied to a fresh
    copy at ``output_path``.
    """
    coll = load_frf(_abs(input_path), output_path=_abs(output_path))
    coll.change_freq_resolution(float(new_resolution))
    path = _close_keep(coll)
    return _describe(path)


@mcp.tool()
def change_freq_span(
    input_path: str,
    output_path: str,
    new_min_freq: Optional[float] = None,
    new_max_freq: Optional[float] = None,
) -> dict:
    """Crop or extend every FRF along the frequency axis.

    ``None`` keeps the existing min/max. Extension to a larger span pads with
    zeros at the appropriate end.
    """
    coll = load_frf(_abs(input_path), output_path=_abs(output_path))
    coll.change_freq_span(new_min_freq=new_min_freq, new_max_freq=new_max_freq)
    path = _close_keep(coll)
    return _describe(path)


@mcp.tool()
def change_sampling_rate(
    input_path: str,
    output_path: str,
    new_sampling_rate: float,
) -> dict:
    """Resample every signal in a ``timeseries`` collection to ``new_sampling_rate`` (Hz)."""
    coll = load_timeseries(_abs(input_path), output_path=_abs(output_path))
    coll.change_sampling_rate(float(new_sampling_rate))
    path = _close_keep(coll)
    return _describe(path)


@mcp.tool()
def change_time_span(
    input_path: str,
    output_path: str,
    new_min_time: Optional[float] = None,
    new_max_time: Optional[float] = None,
) -> dict:
    """Crop or extend every signal in a ``timeseries`` collection along the time axis."""
    coll = load_timeseries(_abs(input_path), output_path=_abs(output_path))
    coll.change_time_span(new_min_time=new_min_time, new_max_time=new_max_time)
    path = _close_keep(coll)
    return _describe(path)


# ── augmentation ──────────────────────────────────────────────────────────────

@mcp.tool()
def add_gaussian_noise(
    input_path: str,
    output_path: str,
    min_amplitude: float = 0.001,
    max_amplitude: float = 0.015,
    sample_fraction: float = 1.0,
) -> dict:
    """Append Gaussian-noise-augmented copies of items in a ``timeseries`` collection.

    For ``sample_fraction`` of items (chosen uniformly at random), a copy
    with additive Gaussian noise is appended with the suffix ``_augmented``.
    Original items are preserved.
    """
    coll = load_timeseries(_abs(input_path), output_path=_abs(output_path))
    coll.AddGaussianNoise(
        min_amplitude=min_amplitude,
        max_amplitude=max_amplitude,
        sample=float(sample_fraction),
    )
    path = _close_keep(coll)
    return _describe(path)


# ── time-domain → frequency-domain ────────────────────────────────────────────

@mcp.tool()
def timeseries_to_frf(
    response_path: str,
    excitation_path: str,
    output_path: str,
    frf_type: str = "H1",
    resp_delay: int = 0,
) -> dict:
    """Compute FRFs from a response/excitation pair of ``timeseries`` collections.

    Both collections must have identical length and matching channel layout.
    The excitation collection must have been built with
    ``method="excitation"``. The resulting :class:`pymodal.frf` keeps both
    inputs as embedded references (roles ``"input"`` and ``"output"``), so
    the produced HDF5 file is fully self-contained.

    ``frf_type`` is one of ``"H1"``, ``"H2"``, ``"Hv"``, ``"vector"``,
    ``"ODS"`` (passed through to pyFRF).
    """
    resp = load_timeseries(_abs(response_path))
    exc = load_timeseries(_abs(excitation_path))
    out = resp.to_FRF(
        excitation=exc,
        FRF_type=frf_type,
        resp_delay=int(resp_delay),
        path=_abs(output_path),
    )
    path = _close_keep(out)
    resp.close()
    exc.close()
    return _describe(path)


# ── SHM indicators ────────────────────────────────────────────────────────────

_INDICATOR_METHODS = {
    "cfdac", "cfdac_a", "fdac",
    "rvac", "rvac_2d", "gac", "m2l",
    "sci", "unsigned_sci", "drq", "aigac",
    "frfrms", "frfsf", "frfsm", "ods_diff", "r2_imag",
}


@mcp.tool()
def compute_indicator(
    indicator: str,
    reference_path: str,
    damaged_path: str,
    output_path: str,
    std: float = 6.0,
) -> dict:
    """Compute a SHM indicator collection from a (reference, damaged) FRF pair.

    Parameters
    ----------
    indicator : str
        Name of the indicator method on :class:`pymodal.frf`. Use
        :func:`list_indicators` to discover the available names.
    reference_path : str
        Path to the baseline / reference FRF collection.
    damaged_path : str
        Path to the FRF collection to compare against the reference. The
        result has one item per item in this collection.
    output_path : str
        Where to write the resulting indicator collection. The file embeds
        both ``reference`` and ``damaged`` as references.
    std : float
        Only used by ``frfsm`` (default 6.0); ignored for other indicators.
    """
    if indicator not in _INDICATOR_METHODS:
        raise ValueError(
            f"Unknown indicator '{indicator}'. "
            f"Available: {sorted(_INDICATOR_METHODS)}"
        )
    ref = load_frf(_abs(reference_path))
    dmg = load_frf(_abs(damaged_path))
    method = getattr(dmg, indicator)
    kwargs = {"path": _abs(output_path)}
    if indicator == "frfsm":
        kwargs["std"] = float(std)
    out = method(ref, **kwargs)
    path = _close_keep(out)
    ref.close()
    dmg.close()
    return _describe(path)


# ── splitting / dataset handoff ───────────────────────────────────────────────

@mcp.tool()
def split_collection(
    path: str,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    kind: str = "frf",
) -> dict:
    """Compute a stratified train/val/test index split for a labelled collection.

    The collection must have labels for stratification to be meaningful. The
    indices are returned as plain Python lists (and are also stored as
    ``train_indices`` / ``val_indices`` / ``test_indices`` on the in-memory
    collection while it is open). Pass ``kind="timeseries"`` to load a
    ``timeseries`` instead of an ``frf``.
    """
    loader = load_frf if kind == "frf" else load_timeseries
    coll = loader(_abs(path))
    train, val, test = coll.split(
        train_frac=float(train_frac),
        val_frac=float(val_frac),
        test_frac=float(test_frac),
        seed=int(seed),
    )
    coll.close(keep=True)
    return {
        "train_indices": list(map(int, train)),
        "val_indices": list(map(int, val)),
        "test_indices": list(map(int, test)),
    }


@mcp.tool()
def torch_dataset_summary(path: str) -> dict:
    """Open the collection at ``path`` as a :class:`pymodal.HDF5Dataset` and report shape stats.

    Confirms the file is consumable by a PyTorch ``DataLoader`` and returns
    the number of samples, the shape and dtype of the first sample, and
    whether labels are present. Does not load the full dataset into memory.
    """
    ds = pymodal.HDF5Dataset(_abs(path))
    n = len(ds)
    if n == 0:
        return {"n_samples": 0, "labelled": False}
    x, y = ds[0]
    return {
        "n_samples": int(n),
        "sample_shape": list(x.shape),
        "sample_dtype": str(x.dtype),
        "label_value_first": (
            float(y.item()) if y.numel() == 1 else y.tolist()
        ),
        "labelled": True,
    }


# ── simulation ────────────────────────────────────────────────────────────────

@mcp.tool()
def synthetic_frf(
    output_path: str,
    min_freq: float,
    max_freq: float,
    resolution: float,
    natural_frequencies: list[float],
    damping_ratios: list[float],
    name: Optional[str] = None,
    label: Optional[float] = None,
) -> dict:
    """Generate a single synthetic FRF and store it as a 1-item ``frf`` collection.

    Uses :func:`pymodal.synthetic_FRF` which sums modal contributions of the
    form ``1 / (1 - (ω/ω_n)² + 2 j ζ ω/ω_n)`` for every ``(ω_n, ζ)`` pair.
    Helpful for smoke-testing pipelines without a real measurement.
    """
    nf = np.asarray(natural_frequencies, dtype=float)
    z = np.asarray(damping_ratios, dtype=float)
    if nf.shape != z.shape:
        raise ValueError(
            "natural_frequencies and damping_ratios must have the same length."
        )
    arr = pymodal.synthetic_FRF(
        min_freq=float(min_freq),
        max_freq=float(max_freq),
        resolution=float(resolution),
        natural_frequencies=nf,
        damping=z,
    )
    arr = np.asarray(arr).reshape((-1, 1, 1))
    coll = pymodal.frf(
        measurements=arr,
        freq_start=float(min_freq),
        freq_resolution=float(resolution),
        name=name,
        labels=[float(label)] if label is not None else None,
        path=_abs(output_path),
    )
    path = _close_keep(coll)
    return _describe(path)


# ── entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the MCP server over stdio.

    Equivalent to ``python -m pymodal.mcp``. Configure your MCP client (Claude
    Desktop, Claude Code, …) to launch this command as the server entrypoint.
    """
    # FastMCP defaults to stdio transport, which is what local MCP clients
    # expect when they spawn the server as a subprocess.
    mcp.run()


if __name__ == "__main__":
    main()
