"""End-to-end smoke test for the pymodal MCP tools.

Drives the FastMCP-registered tool callables directly (so we don't have to
spawn a stdio subprocess) to verify the full happy path:

create timeseries → augment → to_FRF → indicator → split → torch dataset.

Each tool is invoked through ``mcp.call_tool(...)`` so we exercise the same
input-validation + argument-coercion path that an MCP client would.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

import pymodal
from pymodal.mcp import mcp


def _call(name: str, arguments: dict) -> dict:
    """Synchronously invoke an MCP tool and return the parsed JSON payload."""
    async def _go():
        result = await mcp.call_tool(name, arguments)
        # FastMCP returns a tuple of (content_list, structured_payload) on
        # newer versions; fall back to scanning the text content otherwise.
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            return result[1]
        items = result[0] if isinstance(result, tuple) else result
        for c in items:
            text = getattr(c, "text", None)
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {"text": text}
        return {}
    return asyncio.run(_go())


def _save_npy(path: Path, arr: np.ndarray) -> Path:
    np.save(path, arr)
    return path


def test_pipeline(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)

    # 1) Build a tiny synthetic FRF collection from .npy arrays.
    n_freq, n_out = 64, 2
    arrs = [
        (rng.standard_normal((n_freq, n_out, 1)) +
         1j * rng.standard_normal((n_freq, n_out, 1))).astype(np.complex64)
        for _ in range(3)
    ]
    arr_paths = [str(_save_npy(tmp_path / f"frf_{i}.npy", a)) for i, a in enumerate(arrs)]

    out_a = str(tmp_path / "ref.h5")
    info_a = _call("create_frf_collection", {
        "data_paths": arr_paths,
        "output_path": out_a,
        "freq_start": 0.0,
        "freq_resolution": 1.0,
        "labels": [0.0, 0.0, 0.0],
    })
    assert info_a["n_items"] == 3
    assert info_a["item_shape"] == [n_freq, n_out, 1]
    assert info_a["n_domain_axes"] == 1
    assert info_a["domain_axes"][0]["length"] == n_freq

    # 2) Build a perturbed "damaged" collection.
    out_b = str(tmp_path / "dmg.h5")
    arrs_d = [a + 0.5 * rng.standard_normal(a.shape).astype(np.complex64) for a in arrs]
    arr_paths_d = [
        str(_save_npy(tmp_path / f"dfrf_{i}.npy", a)) for i, a in enumerate(arrs_d)
    ]
    _call("create_frf_collection", {
        "data_paths": arr_paths_d,
        "output_path": out_b,
        "freq_start": 0.0,
        "freq_resolution": 1.0,
        "labels": [1.0, 1.0, 1.0],
    })

    # 3) Compute an SCI indicator (0-D) — verifies indicator dispatch and
    #    embedded references survive in the output file.
    out_c = str(tmp_path / "sci.h5")
    info_c = _call("compute_indicator", {
        "indicator": "sci",
        "reference_path": out_a,
        "damaged_path": out_b,
        "output_path": out_c,
    })
    assert info_c["n_items"] == 3
    assert info_c["n_domain_axes"] == 0
    roles = {r["role"] for r in info_c["reference_roles"]}
    assert {"reference", "damaged"}.issubset(roles)

    # 4) Compute a CFDAC indicator (2-D) — verifies higher-rank indicators.
    out_d = str(tmp_path / "cfdac.h5")
    info_d = _call("compute_indicator", {
        "indicator": "cfdac",
        "reference_path": out_a,
        "damaged_path": out_b,
        "output_path": out_d,
    })
    assert info_d["n_domain_axes"] == 2

    # 5) Splitting a labelled collection.
    split = _call("split_collection", {
        "path": out_b,
        "train_frac": 1.0,
        "val_frac": 0.0,
        "test_frac": 0.0,
        "kind": "frf",
    })
    assert sorted(split["train_indices"]) == [0, 1, 2]

    # 6) HDF5Dataset summary.
    summary = _call("torch_dataset_summary", {"path": out_b})
    assert summary["n_samples"] == 3
    assert summary["sample_shape"] == [n_freq, n_out, 1]
    assert summary["labelled"] is True

    # 7) describe_collection on the indicator file.
    info_d_again = _call("describe_collection", {"path": out_d})
    assert info_d_again["n_items"] == 3


def test_synthetic_frf(tmp_path: Path) -> None:
    out = str(tmp_path / "syn.h5")
    info = _call("synthetic_frf", {
        "output_path": out,
        "min_freq": 0.0,
        "max_freq": 10.0,
        "resolution": 0.5,
        "natural_frequencies": [3.0, 7.0],
        "damping_ratios": [0.02, 0.02],
    })
    assert info["n_items"] == 1
    assert info["n_domain_axes"] == 1


def test_list_indicators() -> None:
    info = _call("list_indicators", {})
    for fam in ("0d", "1d", "2d"):
        assert fam in info
    assert "cfdac" in info["2d"]
    assert "sci" in info["0d"]
