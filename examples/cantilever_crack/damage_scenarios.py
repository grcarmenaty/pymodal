"""Cantilever-beam crack scenarios for the damage-diagnosis demo.

Uses the mesh-agnostic primitives in :mod:`pymodal.scenarios`:

* one ``pristine`` :class:`pymodal.Scenario`;
* one cracked scenario per ``(crack_position, crack_depth)`` pair on the
  grid defined by :data:`params.DAMAGE_POSITIONS` and
  :data:`params.DAMAGE_DEPTHS` (default 6 x 4 = 24 cracked classes,
  25 classes total);
* a :func:`pymodal.ParameterVariation` with small Gaussian sigmas on every
  continuous parameter (length, width, thickness, modulus, density,
  damping) that produces realistic intra-class spread without changing the
  damage descriptor;
* a :func:`build_dataset` wrapper that pre-wires the reduced-order modal
  provider in :mod:`cantilever_model`.  Pass a different ``modal_provider``
  to swap in an FEA wrapper.

Sensor / shaker layout helpers (:func:`shaker_input`,
:func:`sensor_outputs`) encode the requested setup: a single transverse
shaker near the fixed end and ten transverse accelerometers spread evenly
along the beam.
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
from cantilever_model import BeamGeometry, compute_frf_matrix  # noqa: E402


# ---------------------------------------------------------------------------
# Scenario callbacks
# ---------------------------------------------------------------------------
def _no_damage(_geom: BeamGeometry) -> None:
    pass


def _crack(position_frac: float, depth_frac: float) -> Callable[[BeamGeometry], None]:
    def _apply(geom: BeamGeometry) -> None:
        geom.crack_position_frac = position_frac
        geom.crack_depth_frac = depth_frac
    return _apply


# ---------------------------------------------------------------------------
# Scenario catalogue
# ---------------------------------------------------------------------------
def cantilever_scenarios(positions: Sequence[float] = P.DAMAGE_POSITIONS,
                          depths: Sequence[float] = P.DAMAGE_DEPTHS
                          ) -> List[pm.Scenario]:
    """Pristine + every (position, depth) crack scenario, dense integer labels."""
    out: List[pm.Scenario] = [
        pm.Scenario(label=0, name='pristine', apply=_no_damage),
    ]
    label = 1
    for p_idx, p in enumerate(positions):
        for d_idx, d in enumerate(depths):
            out.append(pm.Scenario(
                label=label,
                name=f'p{p_idx}_d{d_idx}_x{int(round(p*100)):02d}_h{int(round(d*100)):02d}',
                apply=_crack(p, d),
            ))
            label += 1
    return out


def scenario_meta(scenarios: Sequence[pm.Scenario],
                   positions: Sequence[float] = P.DAMAGE_POSITIONS,
                   depths: Sequence[float] = P.DAMAGE_DEPTHS):
    """Per-scenario metadata arrays for the three diagnostic heads.

    Returns
    -------
    is_damaged : (n,) ndarray of int   - 0 if pristine, 1 otherwise
    pos_frac   : (n,) ndarray of float - x_c / L of the crack (0 if pristine)
    depth_frac : (n,) ndarray of float - d / h of the crack (0 if pristine)
    """
    n = len(scenarios)
    is_dmg = np.zeros(n, dtype=int)
    pos = np.zeros(n, dtype=float)
    depth = np.zeros(n, dtype=float)
    for i, sc in enumerate(scenarios):
        if sc.name == 'pristine':
            continue
        is_dmg[i] = 1
        # name: p{p_idx}_d{d_idx}_xNN_hNN
        parts = sc.name.split('_')
        p_idx = int(parts[0][1:])
        d_idx = int(parts[1][1:])
        pos[i] = positions[p_idx]
        depth[i] = depths[d_idx]
    return is_dmg, pos, depth


# ---------------------------------------------------------------------------
# Default LANL-style parameter variation
# ---------------------------------------------------------------------------
def default_variation() -> pm.ParameterVariation:
    """Small Gaussian sigmas on every continuous beam parameter.

    The crack location and depth are intentionally **not** perturbed: each
    sample of a class shares the same crack but sees a slightly different
    underlying beam (length, width, thickness, modulus, density, damping).
    That is the realistic interpretation of "manufacturing tolerance" and
    keeps the regression labels valid for every sample.
    """
    return pm.ParameterVariation(
        sigmas={
            'L':       0.005,
            'b':       0.005,
            'h':       0.005,
            'E':       0.005,
            'rho':     0.005,
            'damping': 0.10,
        },
        floor_at={'damping': 1.0e-4},
    )


def base_factory() -> BeamGeometry:
    return BeamGeometry()


# ---------------------------------------------------------------------------
# Modal provider + dataset builder
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
    """Cantilever-flavoured wrapper around :func:`pymodal.build_frf_collection`."""
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
def shaker_input(geom: Optional[BeamGeometry] = None):
    """Single transverse shaker at ``x = SHAKER_FRAC * L``."""
    g = geom or BeamGeometry()
    x = P.SHAKER_FRAC * g.L
    return [((x, 0.0, 0.0), P.SHAKER_DIR)]


def sensor_outputs(geom: Optional[BeamGeometry] = None,
                    fracs: Sequence[float] = P.SENSOR_FRACS):
    """Transverse accelerometers at every fraction of the length in ``fracs``."""
    g = geom or BeamGeometry()
    return [((f * g.L, 0.0, 0.0), P.SENSOR_DIR) for f in fracs]
