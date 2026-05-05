"""Damage scenario catalogue and small-variation sampler for the Los Alamos
3-storey benchmark.

A *scenario* is a labelled, named perturbation of the nominal building. The
present module ships 13 scenarios:

* ``pristine``                         (label 0)
* ``s{storey}_c{corner}_removed``      (labels 1..12)

For every scenario a stream of ``BuildingGeometry`` realisations is produced
by :func:`sample`: dimensions, gap and material parameters are perturbed
independently with small Gaussian noise so that each realisation represents
a slightly different physical specimen of the same nominal class.

Generic shape
-------------
The trio ``(BuildingScenario, sample, build_frf_collection)`` is intentionally
mesh-agnostic. To repurpose this workflow for any other structure:

1. Define a new ``BuildingGeometry`` (or replacement) that captures the
   continuous parameters of your geometry plus a damage descriptor.
2. Subclass / construct ``BuildingScenario`` instances that flip the damage
   descriptor for each labelled class.
3. Plug a different ``modal_provider`` into :func:`build_frf_collection` -
   the default uses the cheap reduced-order model in :mod:`reduced_model`,
   but an FEA path through :mod:`frf_extract` (or any callable returning
   modal frequencies / shapes) drops in unchanged.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import params as P  # noqa: E402
from reduced_model import (BuildingGeometry, compute_frf_matrix)  # noqa: E402

Point = Tuple[float, float, float]
Direction = str

# A modal_provider produces FRF data given (geom, freqs, inputs, outputs).
# Default is the reduced-order model; users can swap in :func:`frf_extract.extract_frf`
# wrappers to trade speed for FEA fidelity.
ModalProvider = Callable[..., np.ndarray]


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------
@dataclass
class BuildingScenario:
    """One damage class.

    ``apply`` mutates a fresh :class:`BuildingGeometry` to encode the damage
    (e.g. flip ``column_active[s, c]`` to ``False``). Geometry perturbation
    is left to the variation sampler so that the same scenario can be
    sampled many times.
    """
    label: int
    name: str
    apply: Callable[[BuildingGeometry], None]

    def base_geometry(self) -> BuildingGeometry:
        g = BuildingGeometry()
        g.column_active = g.column_active.copy()
        self.apply(g)
        return g


def _no_damage(_geom: BuildingGeometry) -> None:
    pass


def _remove_column(storey: int, corner: int):
    def _apply(geom: BuildingGeometry) -> None:
        geom.column_active[storey, corner] = False
    return _apply


def los_alamos_scenarios(n_stories: int = P.N_STORIES) -> List[BuildingScenario]:
    """Pristine + every-column-missing scenario list (1 + 4 * n_stories items)."""
    out: List[BuildingScenario] = [
        BuildingScenario(label=0, name="pristine", apply=_no_damage),
    ]
    label = 1
    for s in range(n_stories):
        for c in range(4):
            out.append(BuildingScenario(
                label=label,
                name=f"s{s}_c{c}_removed",
                apply=_remove_column(s, c),
            ))
            label += 1
    return out


# ---------------------------------------------------------------------------
# Small geometric / material variation sampler
# ---------------------------------------------------------------------------
@dataclass
class VariationSpec:
    """Relative Gaussian sigmas for each perturbed quantity."""
    plate_lx: float = 0.005
    plate_ly: float = 0.005
    plate_lz: float = 0.005
    col_lx:   float = 0.010
    col_ly:   float = 0.010
    inter_storey_gap: float = 0.005
    column_inset:     float = 0.010
    young:    float = 0.005
    density:  float = 0.005
    damping:  float = 0.10        # damping varies more than dimensions


def sample(scenario: BuildingScenario,
            seed: int,
            spec: Optional[VariationSpec] = None) -> BuildingGeometry:
    """Draw one perturbed realisation of ``scenario``.

    Each scalar parameter ``x`` is multiplied by ``1 + sigma * N(0, 1)`` with
    ``sigma`` taken from ``spec``. The damage mask is applied last so the
    geometric perturbation never accidentally re-introduces a removed column.
    """
    spec = spec or VariationSpec()
    rng = np.random.default_rng(seed)
    g = BuildingGeometry()
    g.plate_lx          *= 1.0 + spec.plate_lx          * rng.standard_normal()
    g.plate_ly          *= 1.0 + spec.plate_ly          * rng.standard_normal()
    g.plate_lz          *= 1.0 + spec.plate_lz          * rng.standard_normal()
    g.col_lx            *= 1.0 + spec.col_lx            * rng.standard_normal()
    g.col_ly            *= 1.0 + spec.col_ly            * rng.standard_normal()
    g.inter_storey_gap  *= 1.0 + spec.inter_storey_gap  * rng.standard_normal()
    g.column_inset      *= 1.0 + spec.column_inset      * rng.standard_normal()
    g.young             *= 1.0 + spec.young             * rng.standard_normal()
    g.density            *= 1.0 + spec.density           * rng.standard_normal()
    g.damping            *= 1.0 + spec.damping           * rng.standard_normal()
    g.damping = max(g.damping, 1.0e-4)
    g.column_active = g.column_active.copy()
    scenario.apply(g)
    return g


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------
def build_frf_collection(scenarios: Sequence[BuildingScenario],
                          n_per_scenario: int,
                          inputs: Sequence[Tuple[Point, Direction]],
                          outputs: Sequence[Tuple[Point, Direction]],
                          freq_array: np.ndarray,
                          path,
                          spec: Optional[VariationSpec] = None,
                          seed: int = 0,
                          modal_provider: Optional[ModalProvider] = None,
                          progress: bool = False):
    """Generate one big labelled :class:`pymodal.frf` collection.

    Parameters
    ----------
    scenarios : sequence of :class:`BuildingScenario`
    n_per_scenario : int
        Number of small-variation realisations to draw per scenario.
    inputs, outputs : sequence of ((x, y, z), 'X' | 'Y' | 'Z')
        Driving and roving (point, direction) pairs that define the FRF
        matrix shape ``(n_freq, len(outputs), len(inputs))``.
    freq_array : (n_freq,) numpy.ndarray
    path : str or pathlib.Path
        HDF5 destination for the collection.
    spec : :class:`VariationSpec`, optional
    seed : int
        Master seed; the per-sample seed is ``seed * 1_000_003 + global_index``.
    modal_provider : callable, optional
        ``(geom, freqs, inputs, outputs) -> ndarray`` producing the
        ``(n_freq, n_outputs, n_inputs)`` complex FRF for one realisation.
        Defaults to the reduced-order model's :func:`compute_frf_matrix`.
    progress : bool
        Print a one-line tick per scenario.

    Returns
    -------
    pymodal.frf
        Multi-item collection with one item per ``(scenario, sample)`` pair,
        each labelled with ``scenario.label``.
    """
    # Late import - avoids importing torch when only the model is needed.
    from pymodal import frf

    if modal_provider is None:
        def modal_provider(geom, freqs, ins, outs):
            return compute_frf_matrix(freqs, list(ins), list(outs), geom)

    items: List[np.ndarray] = []
    names: List[str] = []
    labels: List[float] = []

    global_idx = 0
    for sc in scenarios:
        for k in range(n_per_scenario):
            geom = sample(sc, seed=seed * 1_000_003 + global_idx, spec=spec)
            H = modal_provider(geom, freq_array, inputs, outputs)
            items.append(H.astype(np.complex64))
            names.append(f"{sc.name}_{k:03d}")
            labels.append(float(sc.label))
            global_idx += 1
        if progress:
            print(f"  [{sc.label:>2d}] {sc.name:<25s}: "
                  f"{n_per_scenario} samples done")

    return frf(measurements=items,
                freq_array=freq_array,
                names=names,
                labels=labels,
                method="SIMO",
                path=path)
