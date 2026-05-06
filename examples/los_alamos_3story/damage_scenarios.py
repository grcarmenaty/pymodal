"""LANL-specific scenario catalogue for the Los Alamos 3-storey benchmark.

The mesh-agnostic plumbing - :class:`pymodal.Scenario`,
:class:`pymodal.ParameterVariation`, :func:`pymodal.sample` and
:func:`pymodal.build_frf_collection` - lives in :mod:`pymodal.scenarios`.
This module only contributes the LANL-specific bits:

* a base-geometry factory (``BuildingGeometry()`` with all column factors
  at 1.0);
* the default :class:`ParameterVariation` for small geometric / material
  perturbations;
* the damage scenario list - one *pristine* scenario plus one
  *progressive thinning* scenario per (storey, corner, level), where each
  level multiplies the thickness of one column by a factor in
  ``DAMAGE_LEVELS``;
* a thin :func:`build_dataset` wrapper that wires the library builder to
  the reduced-order modal provider.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import params as P  # noqa: E402
import pymodal as pm  # noqa: E402
from reduced_model import (BuildingGeometry, compute_frf_matrix)  # noqa: E402

Point = Tuple[float, float, float]
Direction = str

# Column linear index helper: storey s in 0..n_stories-1, corner c in 0..3
def _col_index(storey: int, corner: int) -> int:
    return storey * 4 + corner


# ---------------------------------------------------------------------------
# Damage definition
# ---------------------------------------------------------------------------
DAMAGE_LEVELS: Tuple[float, ...] = (0.75, 0.50, 0.25, 0.10)
"""Section-scale factors used as discrete severity classes.

A factor of ``0.10`` shrinks both column dimensions to 10 % of nominal,
and the lateral stiffness to ``0.10 ** 4 ≈ 1e-4`` of nominal - effectively
a fully damaged column. ``1.00`` would be pristine; the pristine case is
held by its own scenario, so it does not appear here."""


def _no_damage(_geom: BuildingGeometry) -> None:
    pass


def _thin_column(storey: int, corner: int, factor: float) -> Callable[[BuildingGeometry], None]:
    def _apply(geom: BuildingGeometry) -> None:
        geom.column_factor = geom.column_factor.copy()
        geom.column_factor[storey, corner] = factor
    return _apply


# ---------------------------------------------------------------------------
# Scenario catalogue
# ---------------------------------------------------------------------------
def los_alamos_scenarios(n_stories: int = P.N_STORIES,
                          levels: Sequence[float] = DAMAGE_LEVELS
                          ) -> List[pm.Scenario]:
    """Pristine + (storey, corner, level) thinning scenarios.

    Total scenarios = ``1 + 4 * n_stories * len(levels)``.

    Labels are dense integers starting at ``0`` (pristine). For LANL with
    three storeys and four levels there are 49 classes. Auxiliary metadata
    (column index 0..11, severity level index 0..len(levels)-1) for the
    multi-head classifiers is recoverable from the name and from the
    helper :func:`scenario_meta`.
    """
    out: List[pm.Scenario] = [
        pm.Scenario(label=0, name="pristine", apply=_no_damage),
    ]
    label = 1
    for s in range(n_stories):
        for c in range(4):
            for L_idx, factor in enumerate(levels):
                out.append(pm.Scenario(
                    label=label,
                    name=f"s{s}_c{c}_lvl{L_idx}_f{int(round(factor*100)):02d}",
                    apply=_thin_column(s, c, factor),
                ))
                label += 1
    return out


def scenario_meta(scenarios: Sequence[pm.Scenario],
                   levels: Sequence[float] = DAMAGE_LEVELS):
    """Return per-scenario metadata arrays for the three classifier heads.

    Parameters
    ----------
    scenarios : sequence of :class:`pymodal.Scenario`

    Returns
    -------
    is_damaged : (n,) ndarray of int     - 0 if pristine, 1 otherwise
    col_label  : (n,) ndarray of int     - column index 0..11 (or -1 if pristine)
    sev_label  : (n,) ndarray of int     - severity level 0..len(levels)-1 (or -1)
    factor     : (n,) ndarray of float   - section-scale factor of the
                                            thinned column ``∈ [0, 1]``.
                                            Pristine = 1.0. This is the
                                            *recommended regression target*
                                            because the four LANL damage
                                            levels (0.75, 0.50, 0.25, 0.10)
                                            are evenly spread, so a regressor
                                            trained on it does not collapse
                                            the way one trained directly on
                                            ``loss = 1 - factor**4`` would
                                            (three of those four loss values
                                            sit near 1).
    rig_loss   : (n,) ndarray of float   - convenience: ``1 - factor**4``,
                                            the loss of rigidity in [0, 1].
                                            Reported at evaluation time; not
                                            recommended as a training target.
    """
    is_dmg = np.zeros(len(scenarios), dtype=int)
    col = -np.ones(len(scenarios), dtype=int)
    sev = -np.ones(len(scenarios), dtype=int)
    factor = np.ones(len(scenarios), dtype=float)
    rig = np.zeros(len(scenarios), dtype=float)
    for i, sc in enumerate(scenarios):
        if sc.name == "pristine":
            continue
        is_dmg[i] = 1
        # name: f"s{s}_c{c}_lvl{L_idx}_f{...}"
        parts = sc.name.split('_')
        s = int(parts[0][1:])
        c = int(parts[1][1:])
        L = int(parts[2][3:])
        col[i] = _col_index(s, c)
        sev[i] = L
        factor[i] = levels[L]
        rig[i] = 1.0 - levels[L] ** 4
    return is_dmg, col, sev, factor, rig


# ---------------------------------------------------------------------------
# Default LANL parameter variation
# ---------------------------------------------------------------------------
def default_variation() -> pm.ParameterVariation:
    """Small Gaussian sigmas applied to each :class:`BuildingGeometry`
    realisation. Roughly 0.5 % on geometric quantities, 1 % on column
    cross-section dimensions (which directly perturb modal frequencies),
    and 10 % on damping."""
    return pm.ParameterVariation(
        sigmas={
            "plate_lx":         0.005,
            "plate_ly":         0.005,
            "plate_lz":         0.005,
            "col_lx":           0.010,
            "col_ly":           0.010,
            "inter_storey_gap": 0.005,
            "column_gap":       0.05,
            "young":            0.005,
            "density":          0.005,
            "damping":          0.10,
        },
        floor_at={"damping": 1.0e-4, "column_gap": 1.0e-5},
    )


def base_factory() -> BuildingGeometry:
    """Factory returning a fresh nominal :class:`BuildingGeometry`."""
    return BuildingGeometry()


# ---------------------------------------------------------------------------
# High-level dataset builder (pre-wired to the reduced-order model)
# ---------------------------------------------------------------------------
def _reduced_order_provider(geom, freq_array, inputs, outputs):
    return compute_frf_matrix(freq_array, list(inputs), list(outputs), geom)


def build_dataset(scenarios: Sequence[pm.Scenario],
                   n_per_scenario: int,
                   inputs: Sequence,
                   outputs: Sequence,
                   freq_array: np.ndarray,
                   path,
                   variation: Optional[pm.ParameterVariation] = None,
                   seed: int = 0,
                   progress: bool = False,
                   modal_provider=None):
    """LANL-flavoured wrapper around :func:`pymodal.build_frf_collection`.

    Defaults to the cheap reduced-order modal provider; pass
    ``modal_provider`` to swap in an FEA wrapper without changing anything
    else.
    """
    return pm.build_frf_collection(
        scenarios=scenarios,
        n_per_scenario=n_per_scenario,
        modal_provider=modal_provider or _reduced_order_provider,
        freq_array=freq_array,
        inputs=inputs,
        outputs=outputs,
        path=path,
        base_factory=base_factory,
        variation=variation if variation is not None else default_variation(),
        seed=seed,
        progress=progress,
    )


# ---------------------------------------------------------------------------
# Sensor / shaker layout helpers
# ---------------------------------------------------------------------------
def shaker_input(geom: Optional[BuildingGeometry] = None):
    """Single shaker at the centre of the base plate, in the rail direction."""
    g = geom or BuildingGeometry()
    cx, cy = g.plate_centroid
    z = g.plate_z_centre(0)
    return [((cx, cy, z), g.rail_direction)]


def column_intersection_outputs(geom: Optional[BuildingGeometry] = None,
                                  directions: Sequence[str] = ('X', 'Y')):
    """Accelerometers at every column-plate intersection.

    By default each intersection carries two channels: ``X`` and ``Y``
    (tri-axial-like minus the out-of-plane axis the model does not
    represent). ``Y`` alone is unable to disambiguate mirror-image
    damage classes when the shaker sits at the plate centroid, because
    the Y-component of the response is invariant under the ``X -> -X``
    reflection that swaps left and right columns. Adding ``X`` breaks
    the symmetry and lets the localisation head separate them.

    ``(n_stories + 1) * 4 * len(directions)`` channels total.
    """
    g = geom or BuildingGeometry()
    out = []
    for k in range(g.n_stories + 1):
        z = g.plate_z_centre(k)
        for xc, yc in g.column_attachment_points():
            for d in directions:
                out.append(((xc, yc, z), d))
    return out
