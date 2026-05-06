"""Generic scenario / dataset-building primitives for damage-sensitive
collections.

This module is intentionally **mesh-agnostic**: it does not know what a
"building", a "column" or a "geometry" is. It only knows that a *scenario*
is a labelled, named callback that mutates an opaque "geometry" object, and
that an "FRF batch" is built by repeatedly perturbing such a geometry,
asking a user-supplied callable for the corresponding FRF tensor, and
appending each result as one labelled item of a :class:`pymodal.frf`
collection.

Two pieces of glue are exposed:

:class:`Scenario`
    ``(label, name, apply)`` triple. ``apply(geom)`` mutates the geometry
    in place and is run after the parameter perturbation, so the perturbed
    realisation always carries the requested damage descriptor.

:class:`ParameterVariation`
    Mapping ``attribute_name -> relative sigma``. Each attribute is
    multiplied by ``1 + sigma * N(0, 1)``, allowing any continuous
    parameter to be jittered without the library having to know what the
    parameters mean.

:func:`sample`
    Single perturbed realisation of a scenario.

:func:`build_frf_collection`
    Loop over ``(scenario, sample_index)``, ask a ``modal_provider`` for
    the FRF tensor of shape ``(n_freq, n_outputs, n_inputs)``, and persist
    everything to one HDF5-backed :class:`pymodal.frf` collection.

:func:`closest_node` / :func:`load_nodes_json`
    Helpers for resolving user-supplied ``(x, y, z)`` coordinates against
    a precomputed mesh node list - useful when the modal provider runs FEA
    on a fixed mesh.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Scenario protocol
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    """One labelled damage class.

    ``apply`` mutates a freshly constructed geometry in place; pure /
    parameterless ``apply`` callables (``lambda g: None``) represent the
    pristine class.
    """
    label: int
    name: str
    apply: Callable[[Any], None] = field(default=lambda _g: None)


# ---------------------------------------------------------------------------
# Parameter perturbation
# ---------------------------------------------------------------------------
@dataclass
class ParameterVariation:
    """Per-attribute relative Gaussian sigmas.

    Every entry of ``sigmas`` is the standard deviation of a multiplicative
    perturbation: ``geom.attr *= 1 + sigma * randn()``. Attributes that do
    not appear in ``sigmas`` are left untouched.
    """
    sigmas: Mapping[str, float] = field(default_factory=dict)
    floor_at: Optional[Mapping[str, float]] = None

    def apply(self, geom: Any, rng: np.random.Generator) -> None:
        for name, sigma in self.sigmas.items():
            current = getattr(geom, name)
            current = current * (1.0 + sigma * rng.standard_normal())
            if self.floor_at is not None and name in self.floor_at:
                current = max(current, self.floor_at[name])
            setattr(geom, name, current)


# ---------------------------------------------------------------------------
# Single-realisation sampler
# ---------------------------------------------------------------------------
def sample(scenario: Scenario,
            seed: int,
            base_factory: Callable[[], Any],
            variation: Optional[ParameterVariation] = None) -> Any:
    """Draw one realisation of ``scenario``.

    ``base_factory`` returns a freshly constructed geometry (the user knows
    its type; the library does not). The variation is applied first so the
    scenario callback (which usually flips a damage flag or scales a
    section) is never undone by the perturbation.
    """
    rng = np.random.default_rng(seed)
    geom = base_factory()
    if variation is not None:
        variation.apply(geom, rng)
    scenario.apply(geom)
    return geom


# ---------------------------------------------------------------------------
# Modal provider type alias
# ---------------------------------------------------------------------------
# (geom, freq_array, inputs, outputs) -> ndarray with shape
# (n_freq, n_outputs, n_inputs), complex.
ModalProvider = Callable[[Any, np.ndarray, Sequence, Sequence], np.ndarray]


# ---------------------------------------------------------------------------
# Dataset assembler
# ---------------------------------------------------------------------------
def build_frf_collection(scenarios: Sequence[Scenario],
                          n_per_scenario: int,
                          modal_provider: ModalProvider,
                          freq_array: np.ndarray,
                          inputs: Sequence,
                          outputs: Sequence,
                          path,
                          base_factory: Callable[[], Any],
                          variation: Optional[ParameterVariation] = None,
                          seed: int = 0,
                          progress: bool = False,
                          method: str = "SIMO"):
    """Assemble one labelled multi-item :class:`pymodal.frf` collection.

    Parameters
    ----------
    scenarios : sequence of :class:`Scenario`
    n_per_scenario : int
        Number of perturbed realisations per scenario.
    modal_provider : callable
        ``(geom, freq_array, inputs, outputs) -> ndarray`` returning the
        ``(n_freq, n_outputs, n_inputs)`` complex FRF tensor for one
        realisation.
    freq_array : (n_freq,) numpy.ndarray
    inputs, outputs : sequence
        Driver and roving descriptors. The library does not interpret them
        - they are passed through to ``modal_provider`` verbatim, which is
        what allows the same builder to drive a reduced-order model, an
        FEA wrapper, or anything else.
    path : str or pathlib.Path
        HDF5 destination.
    base_factory : callable
        ``() -> geom`` factory returning a fresh nominal geometry.
    variation : :class:`ParameterVariation`, optional
    seed : int
        Master seed; the per-sample seed is ``seed * 1_000_003 + index``.
    progress : bool
        Print one line per scenario.
    method : str
        ``method`` argument forwarded to :class:`pymodal.frf`.
    """
    # Late import - keeps pymodal.scenarios importable without h5py / pytorch
    from pymodal import frf

    items: List[np.ndarray] = []
    names: List[str] = []
    labels: List[float] = []

    global_idx = 0
    for sc in scenarios:
        for k in range(n_per_scenario):
            geom = sample(sc,
                           seed=seed * 1_000_003 + global_idx,
                           base_factory=base_factory,
                           variation=variation)
            H = modal_provider(geom, freq_array, inputs, outputs)
            items.append(np.asarray(H, dtype=np.complex64))
            names.append(f"{sc.name}_{k:03d}")
            labels.append(float(sc.label))
            global_idx += 1
        if progress:
            print(f"  [{sc.label:>2d}] {sc.name:<28s}: {n_per_scenario} samples")

    return frf(measurements=items,
                freq_array=freq_array,
                names=names,
                labels=labels,
                method=method,
                path=path)


# ---------------------------------------------------------------------------
# Mesh-node helpers (useful when the modal provider runs FEA on a fixed mesh)
# ---------------------------------------------------------------------------
def load_nodes_json(path) -> Tuple[np.ndarray, np.ndarray]:
    """Read a ``{node_id: [x, y, z]}`` JSON file and return ``(ids, coords)``.

    Compatible with the file produced by the example's ``salome_build.py``
    but useful for any mesh.
    """
    with open(path) as fp:
        raw = json.load(fp)
    ids = np.array(sorted(int(k) for k in raw), dtype=np.int64)
    coords = np.array([raw[str(int(n))] for n in ids], dtype=float)
    return ids, coords


def closest_node(point: Tuple[float, float, float],
                  node_ids: np.ndarray,
                  coords: np.ndarray) -> Tuple[int, Tuple[float, float, float]]:
    """Return ``(node_id, actual_xyz)`` of the mesh node closest to ``point``."""
    d2 = np.sum((coords - np.asarray(point, dtype=float)) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return int(node_ids[idx]), tuple(coords[idx].tolist())
