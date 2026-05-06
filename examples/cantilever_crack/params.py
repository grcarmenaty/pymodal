"""Geometry, material, and analysis parameters for the cantilever beam
with a single simulated crack.

All lengths are in metres, masses in kilograms, frequencies in hertz. The
defaults describe a thin steel cantilever typical of an SHM lab specimen
(0.5 m long, 25 mm wide, 5 mm thick), with the first four bending modes
in the 0-700 Hz range.

Layout convention
-----------------
* the beam axis runs along ``x``, with the **fixed end at x = 0** and the
  free tip at ``x = BEAM_L``;
* the cross-section is rectangular, ``BEAM_B`` along ``y`` and ``BEAM_H``
  along ``z`` (the crack is cut into the ``z`` face, so the soft bending
  direction is also ``z``);
* damage is described by a single crack at ``x_c = crack_position_frac *
  BEAM_L`` with relative depth ``d / h = crack_depth_frac``. Both
  parameters live on :class:`cantilever_model.BeamGeometry`.
"""

from __future__ import annotations
from typing import Tuple


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
BEAM_L: float = 0.500        # m, length
BEAM_B: float = 0.025        # m, width  (along y)
BEAM_H: float = 0.005        # m, height (along z, the cracked face)
N_ELEMENTS: int = 60         # FEM discretisation along the length

# ---------------------------------------------------------------------------
# Material (mild steel)
# ---------------------------------------------------------------------------
E:   float = 210.0e9         # Pa
NU:  float = 0.30            # not used by the Bernoulli model, kept for FEA
RHO: float = 7800.0          # kg / m^3
MODAL_DAMPING: float = 0.010 # uniform modal damping ratio (broader peaks
                             # so finite-resolution shifts are resolvable)

# ---------------------------------------------------------------------------
# Default analysis
# ---------------------------------------------------------------------------
F_MIN:  float = 1.0          # Hz
F_MAX:  float = 700.0        # captures the first four bending modes
F_STEP: float = 0.5          # 1399 freq points; resolves sub-Hz peak shifts

# ---------------------------------------------------------------------------
# Excitation / sensor layout
# ---------------------------------------------------------------------------
SHAKER_FRAC: float = 0.05    # x / L of the shaker, transverse direction (z)
SENSOR_FRACS: Tuple[float, ...] = (
    0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00,
)  # ten accelerometers along the beam, transverse direction
SHAKER_DIR: str = 'Z'
SENSOR_DIR: str = 'Z'

# ---------------------------------------------------------------------------
# Damage grid (cracks)
# ---------------------------------------------------------------------------
DAMAGE_POSITIONS: Tuple[float, ...] = (0.10, 0.25, 0.40, 0.55, 0.70, 0.85)
"""Crack positions ``x_c / L``. Span 0.10..0.85 so the regression target
is well-spread for the localisation head."""

DAMAGE_DEPTHS: Tuple[float, ...] = (0.10, 0.25, 0.40, 0.55)
"""Relative crack depths ``d / h``. Spread between minor (10 %) and severe
(55 %) damage."""
