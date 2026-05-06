"""Reduced-order Bernoulli-Euler FEM of a cantilever beam with one crack.

The beam is discretised into ``N_ELEMENTS`` equal-length 2-node, 4-DOF
Euler-Bernoulli beam elements, each with the standard consistent stiffness
and mass matrices.  The crack is modelled with the *broken-section*
approximation: a single element at ``x_c = crack_position_frac * L`` has its
flexural stiffness ``EI`` scaled by ``(1 - d/h)**3``, the cube of the
remaining ligament ratio.  This gives realistic frequency shifts as a
function of crack location and depth - small near the free end, large near
the fixed end - and is the standard reduced-order surrogate used in modal
SHM benchmarks.

Cantilever boundary conditions are applied by deleting the two DOFs at the
fixed node (transverse displacement and rotation at ``x = 0``).  The mode
shapes are mass-normalised, the modal damping is uniform, and the FRF
follows the standard modal-superposition kernel.

Points are mapped to participation vectors via Hermite cubic shape
functions, so the transverse FRF can be evaluated at any continuous
``x \in [0, L]`` rather than only at mesh nodes.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import params as P  # noqa: E402

Point = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Geometry container
# ---------------------------------------------------------------------------
@dataclass
class BeamGeometry:
    """All scalar parameters defining one realisation of the cantilever.

    ``crack_depth_frac = 0.0`` is pristine.  When ``crack_depth_frac > 0``
    the element containing ``x_c = crack_position_frac * L`` has its
    flexural stiffness reduced by ``(1 - crack_depth_frac) ** 3``.
    """

    L: float = P.BEAM_L
    b: float = P.BEAM_B
    h: float = P.BEAM_H
    n_elements: int = P.N_ELEMENTS
    E: float = P.E
    rho: float = P.RHO
    damping: float = P.MODAL_DAMPING
    crack_position_frac: float = 0.0   # x_c / L
    crack_depth_frac: float = 0.0      # d / h, 0.0 = pristine

    # ---- helpers -----------------------------------------------------------
    @property
    def le(self) -> float:
        return self.L / self.n_elements

    @property
    def n_nodes(self) -> int:
        return self.n_elements + 1

    @property
    def n_dof_total(self) -> int:
        return 2 * self.n_nodes      # (w, theta) per node

    @property
    def crack_x(self) -> float:
        return self.crack_position_frac * self.L


# ---------------------------------------------------------------------------
# Element matrices
# ---------------------------------------------------------------------------
def _element_K_template(le: float) -> np.ndarray:
    """4 x 4 dimensionless template; multiply by ``EI`` to get the element K."""
    return np.array([
        [12.0,  6 * le,        -12.0,  6 * le      ],
        [6 * le, 4 * le ** 2,  -6 * le, 2 * le ** 2],
        [-12.0, -6 * le,        12.0,  -6 * le     ],
        [6 * le, 2 * le ** 2,  -6 * le, 4 * le ** 2],
    ]) / le ** 3


def _element_M(le: float, rhoA: float) -> np.ndarray:
    """4 x 4 consistent-mass matrix."""
    return rhoA * le / 420.0 * np.array([
        [156.0,   22 * le,        54.0,   -13 * le      ],
        [22 * le, 4 * le ** 2,    13 * le, -3 * le ** 2 ],
        [54.0,    13 * le,        156.0,  -22 * le      ],
        [-13 * le, -3 * le ** 2,  -22 * le, 4 * le ** 2 ],
    ])


# ---------------------------------------------------------------------------
# Global K, M assembly with crack
# ---------------------------------------------------------------------------
def stiffness_and_mass(geom: BeamGeometry) -> Tuple[np.ndarray, np.ndarray]:
    le = geom.le
    A = geom.b * geom.h
    I0 = geom.b * geom.h ** 3 / 12.0
    rhoA = geom.rho * A
    EI0 = geom.E * I0

    cracked_elem = -1
    if geom.crack_depth_frac > 0.0 and 0.0 <= geom.crack_x < geom.L:
        cracked_elem = min(int(geom.crack_x / le), geom.n_elements - 1)

    n = geom.n_dof_total
    K = np.zeros((n, n))
    M = np.zeros((n, n))
    Ke_template = _element_K_template(le)
    Me = _element_M(le, rhoA)

    for e in range(geom.n_elements):
        if e == cracked_elem:
            EI_e = EI0 * (1.0 - geom.crack_depth_frac) ** 3
        else:
            EI_e = EI0
        Ke = EI_e * Ke_template
        idx = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
        for i, gi in enumerate(idx):
            for j, gj in enumerate(idx):
                K[gi, gj] += Ke[i, j]
                M[gi, gj] += Me[i, j]
    return K, M


def cantilever_dof_indices(geom: BeamGeometry) -> np.ndarray:
    """DOFs remaining after applying the fixed BC at ``x = 0``."""
    return np.arange(2, geom.n_dof_total)


def modes(geom: BeamGeometry) -> Tuple[np.ndarray, np.ndarray]:
    """Mass-normalised modes of the cantilever.

    Returns ``(freqs_hz, V_full)`` where ``V_full`` is shaped
    ``(n_dof_total, n_modes)`` with the fixed DOFs zero-padded so that
    Hermite-shape-function lookups indexed by global DOF still work.
    """
    K, M = stiffness_and_mass(geom)
    free = cantilever_dof_indices(geom)
    K_f = K[np.ix_(free, free)]
    M_f = M[np.ix_(free, free)]
    w2, V = eigh(K_f, M_f)
    w2 = np.clip(w2, 0.0, None)
    freqs = np.sqrt(w2) / (2.0 * np.pi)
    V_full = np.zeros((geom.n_dof_total, V.shape[1]))
    V_full[free, :] = V
    return freqs, V_full


# ---------------------------------------------------------------------------
# Hermite shape functions for point -> DOF participation
# ---------------------------------------------------------------------------
def _hermite(le: float, xi: float) -> np.ndarray:
    """Hermite cubic shape functions for transverse displacement.

    ``xi`` is the local coordinate ``(x - x_e) / le`` with values in [0, 1].
    Returns ``[N1, N2, N3, N4]`` for ``[w_left, theta_left, w_right, theta_right]``.
    """
    return np.array([
        1 - 3 * xi ** 2 + 2 * xi ** 3,
        le * (xi - 2 * xi ** 2 + xi ** 3),
        3 * xi ** 2 - 2 * xi ** 3,
        le * (-xi ** 2 + xi ** 3),
    ])


def point_to_dof_vector(point: Point,
                         direction: str,
                         geom: BeamGeometry) -> np.ndarray:
    """Map ``(x, y, z) + direction`` to a ``(n_dof_total,)`` participation vector.

    The 1-D Bernoulli model only has transverse DOFs, so excitations or
    measurements in directions other than ``Z`` produce a zero vector.  The
    transverse case interpolates with cubic Hermite shape functions, so any
    continuous ``x in [0, L]`` works (not just mesh nodes).
    """
    direction = direction.upper()
    if direction != 'Z':
        return np.zeros(geom.n_dof_total)
    x = float(point[0])
    x = max(0.0, min(x, geom.L))
    le = geom.le
    e = min(int(x / le), geom.n_elements - 1)
    xi = (x - e * le) / le
    N = _hermite(le, xi)
    b = np.zeros(geom.n_dof_total)
    idx = [2 * e, 2 * e + 1, 2 * (e + 1), 2 * (e + 1) + 1]
    for i, gi in enumerate(idx):
        b[gi] = N[i]
    return b


# ---------------------------------------------------------------------------
# FRF computation (modal superposition, accelerance in mm / s^2 / N)
# ---------------------------------------------------------------------------
def compute_frf(freq_array: np.ndarray,
                 input_xyz: Point, input_dir: str,
                 output_xyz: Point, output_dir: str,
                 geom: BeamGeometry,
                 damping: Optional[float] = None) -> np.ndarray:
    if damping is None:
        damping = geom.damping
    freqs, V = modes(geom)
    omega = 2.0 * np.pi * freq_array
    omega_n = 2.0 * np.pi * freqs
    b = point_to_dof_vector(input_xyz, input_dir, geom)
    c = point_to_dof_vector(output_xyz, output_dir, geom)
    bphi = V.T @ b
    cphi = V.T @ c
    H_acc = np.zeros_like(omega, dtype=complex)
    for r in range(len(freqs)):
        num = cphi[r] * bphi[r]
        if omega_n[r] < 1.0e-3:
            H_acc += num
        else:
            den = (omega_n[r] ** 2 - omega ** 2) \
                  + 2j * damping * omega_n[r] * omega
            H_acc += -omega ** 2 * num / den
    return H_acc * 1.0e3                           # m/s^2/N -> mm/s^2/N


def compute_frf_matrix(freq_array: np.ndarray,
                        inputs: Sequence,
                        outputs: Sequence,
                        geom: BeamGeometry,
                        damping: Optional[float] = None) -> np.ndarray:
    """Vectorised version: every ``(output, input)`` FRF in one pass.

    Returns a ``(n_freq, n_outputs, n_inputs)`` complex array, ready as a
    single :class:`pymodal.frf` item.
    """
    if damping is None:
        damping = geom.damping
    freqs, V = modes(geom)
    omega = 2.0 * np.pi * freq_array
    omega_n = 2.0 * np.pi * freqs

    B = np.stack([point_to_dof_vector(p, d, geom) for p, d in inputs], axis=1)
    C = np.stack([point_to_dof_vector(p, d, geom) for p, d in outputs], axis=1)
    bphi = V.T @ B
    cphi = V.T @ C

    is_rigid = omega_n < 1.0e-3
    is_elastic = ~is_rigid
    elastic = np.zeros((len(omega), C.shape[1], B.shape[1]), dtype=complex)
    if is_elastic.any():
        wn = omega_n[is_elastic]
        b_e = bphi[is_elastic]
        c_e = cphi[is_elastic]
        den = (wn[:, None] ** 2 - omega[None, :] ** 2) \
              + 2j * damping * wn[:, None] * omega[None, :]
        kernel = -omega[None, :] ** 2 / den
        elastic = np.einsum('ri,rj,rk->kij', c_e, b_e, kernel)
    rigid = np.zeros((C.shape[1], B.shape[1]), dtype=complex)
    if is_rigid.any():
        rigid = np.einsum('ri,rj->ij', cphi[is_rigid], bphi[is_rigid])
    return (elastic + rigid[None, :, :]) * 1.0e3
