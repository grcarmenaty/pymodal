"""Reduced-order modal model of the Los Alamos 3-storey benchmark.

The full FEA pipeline in :mod:`salome_build` and :mod:`frf_extract` is the
high-fidelity workhorse, but generating hundreds of FRFs per damage scenario
through Code_Aster is prohibitively slow for a notebook demo (each modal
analysis is a few seconds and each harmonic sweep tens of seconds).

This module provides a cheap, physics-based alternative: every floor plate
is treated as a rigid body with three in-plane DOFs ``(x, y, theta_z)`` at
the plate centroid, columns are modelled as Euler-Bernoulli fixed-fixed
beams contributing translational stiffness at their footprint, and the base
plate is grounded. The resulting stiffness matrix is anisotropic and
torsionally coupled, so removing any single column breaks symmetry in a
class-distinctive way.

The :class:`BuildingGeometry` dataclass captures every continuous parameter
(plate / column dimensions, gap, material) and a boolean
``column_active[s, c]`` mask describing which of the four columns at each
storey is intact. This is what gets perturbed to manufacture the small
geometric variations in :mod:`damage_scenarios`.

The same FRF call signature - ``input_xyz, input_dir, output_xyz, output_dir``
- as the FEA workflow is preserved, so a notebook that talks to
:func:`compute_frf` here can be repointed at :func:`frf_extract.extract_frf`
unchanged for higher-fidelity runs on any other mesh.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import eigh

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import params as P  # noqa: E402

Point = Tuple[float, float, float]
Direction = str  # 'X', 'Y' or 'Z'


# ---------------------------------------------------------------------------
# Geometry container
# ---------------------------------------------------------------------------
@dataclass
class BuildingGeometry:
    """All scalar parameters for one realisation of the building.

    Defaults pull from :mod:`params` so the unperturbed geometry matches the
    nominal LANL 3-storey benchmark. Every numeric field can be perturbed
    independently to manufacture small geometric variations.

    ``column_active`` is a ``(n_stories, 4)`` boolean mask. Setting any entry
    to ``False`` removes that column from the stiffness assembly, which is
    how the damage scenarios are encoded.
    """

    plate_lx: float = P.PLATE_LX
    plate_ly: float = P.PLATE_LY
    plate_lz: float = P.PLATE_LZ
    col_lx: float = P.COL_LX
    col_ly: float = P.COL_LY
    inter_storey_gap: float = P.INTER_STOREY_GAP
    column_inset: float = P.COLUMN_INSET
    n_stories: int = P.N_STORIES
    young: float = P.ALU_E
    poisson: float = P.ALU_NU
    density: float = P.ALU_RHO
    damping: float = P.MODAL_DAMPING
    column_active: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.column_active is None:
            self.column_active = np.ones((self.n_stories, 4), dtype=bool)
        else:
            self.column_active = np.asarray(self.column_active, dtype=bool)

    # ---- coordinate helpers ------------------------------------------------
    @property
    def storey_height(self) -> float:
        return self.plate_lz + self.inter_storey_gap

    @property
    def plate_centroid(self) -> Tuple[float, float]:
        return self.plate_lx / 2.0, self.plate_ly / 2.0

    def plate_z_centre(self, k: int) -> float:
        """Mid-thickness z of plate index ``k`` (0 = base, n_stories = roof)."""
        return k * self.storey_height + self.plate_lz / 2.0

    def column_centres(self) -> np.ndarray:
        """``(4, 2)`` array of column-centre ``(x, y)`` in plate coordinates."""
        x_lo = self.column_inset + self.col_lx / 2.0
        y_lo = self.column_inset + self.col_ly / 2.0
        x_hi = self.plate_lx - self.column_inset - self.col_lx / 2.0
        y_hi = self.plate_ly - self.column_inset - self.col_ly / 2.0
        return np.array([(x_lo, y_lo),
                          (x_hi, y_lo),
                          (x_lo, y_hi),
                          (x_hi, y_hi)])


# ---------------------------------------------------------------------------
# Stiffness and mass matrices
# ---------------------------------------------------------------------------
def _column_lateral_stiffnesses(geom: BuildingGeometry) -> Tuple[float, float]:
    """Translational stiffness of one fixed-fixed column in the X and Y
    directions (force per unit relative end displacement)."""
    L = geom.storey_height
    # I_yy used for bending in the X-Z plane (deflection in x, bending about y)
    I_yy = geom.col_ly * geom.col_lx ** 3 / 12.0
    I_xx = geom.col_lx * geom.col_ly ** 3 / 12.0
    kx = 12.0 * geom.young * I_yy / L ** 3
    ky = 12.0 * geom.young * I_xx / L ** 3
    return kx, ky


def stiffness_matrix(geom: BuildingGeometry) -> np.ndarray:
    """Assemble the ``3 n_stories x 3 n_stories`` lateral stiffness matrix.

    DOF ordering: ``[x_1, y_1, theta_1, x_2, y_2, theta_2, ...]`` where
    floor index 1..n_stories runs from the bottom free plate to the roof.
    """
    n = geom.n_stories
    K = np.zeros((3 * n, 3 * n))
    cx, cy = geom.plate_centroid
    centres = geom.column_centres()
    kx, ky = _column_lateral_stiffnesses(geom)
    K_local = np.diag([kx, ky])

    for s in range(n):                               # 0-indexed storey, 0 is bottom
        for c, (xc_abs, yc_abs) in enumerate(centres):
            if not geom.column_active[s, c]:
                continue
            xc = xc_abs - cx
            yc = yc_abs - cy
            T = np.array([[1.0, 0.0, -yc],
                           [0.0, 1.0,  xc]])         # 2x3: (x,y,theta) -> (u_x,u_y)
            top = slice(3 * s, 3 * s + 3)
            K_top_top = T.T @ K_local @ T
            K[top, top] += K_top_top
            if s > 0:
                bot = slice(3 * (s - 1), 3 * (s - 1) + 3)
                cross = T.T @ K_local @ T
                K[top, bot] -= cross
                K[bot, top] -= cross
                K[bot, bot] += cross
    return K


def mass_matrix(geom: BuildingGeometry) -> np.ndarray:
    """Lumped mass / polar-inertia matrix matching :func:`stiffness_matrix`."""
    n = geom.n_stories
    M = np.zeros((3 * n, 3 * n))
    m = geom.density * geom.plate_lx * geom.plate_ly * geom.plate_lz
    J = m * (geom.plate_lx ** 2 + geom.plate_ly ** 2) / 12.0
    for s in range(n):
        M[3 * s,     3 * s]     = m
        M[3 * s + 1, 3 * s + 1] = m
        M[3 * s + 2, 3 * s + 2] = J
    return M


def modes(geom: BuildingGeometry):
    """Solve the generalised eigenvalue problem ``K v = w**2 M v``.

    Returns ``(freqs_hz, V, M)`` where ``V`` is mass-normalised
    (``V.T @ M @ V == I``) and ``freqs_hz`` is sorted ascending.
    """
    K = stiffness_matrix(geom)
    M = mass_matrix(geom)
    w2, V = eigh(K, M)
    w2 = np.clip(w2, 0.0, None)
    freqs = np.sqrt(w2) / (2.0 * np.pi)
    return freqs, V, M


# ---------------------------------------------------------------------------
# Point-on-mesh -> generalized DOF vector
# ---------------------------------------------------------------------------
def point_to_dof_vector(point: Point,
                         direction: Direction,
                         geom: BuildingGeometry) -> np.ndarray:
    """Map a point/direction on the building to its participation vector in
    the 9-DOF model.

    A horizontal force at ``(x, y, z)`` in direction ``X`` projects onto the
    floor centroid as ``b = (1, 0, -(y - y_cg))`` because of the rigid-plate
    kinematics ``u_x = x_cg - r_y * theta``. Similarly an output reading in
    ``X`` extracts the same combination.
    """
    direction = direction.upper().strip()
    if direction not in ('X', 'Y', 'Z'):
        raise ValueError("direction must be 'X', 'Y' or 'Z'")
    x, y, z = point
    cx, cy = geom.plate_centroid
    floor_centres = np.array([geom.plate_z_centre(k + 1)
                               for k in range(geom.n_stories)])
    floor_idx = int(np.argmin(np.abs(floor_centres - z)))
    rx = x - cx
    ry = y - cy
    b = np.zeros(3 * geom.n_stories)
    if direction == 'X':
        b[3 * floor_idx]     = 1.0
        b[3 * floor_idx + 2] = -ry
    elif direction == 'Y':
        b[3 * floor_idx + 1] = 1.0
        b[3 * floor_idx + 2] = rx
    # Z direction has no projection in this in-plane model -> b stays zero
    return b


# ---------------------------------------------------------------------------
# FRF computation (modal superposition, accelerance in mm/s^2/N)
# ---------------------------------------------------------------------------
def compute_frf(freq_array: np.ndarray,
                 input_xyz: Point,
                 input_dir: Direction,
                 output_xyz: Point,
                 output_dir: Direction,
                 geom: BuildingGeometry,
                 damping: Optional[float] = None) -> np.ndarray:
    """Compute the accelerance H_acc(omega) = -omega^2 * H_disp(omega)
    for a unit harmonic force at ``input_xyz`` direction ``input_dir`` and a
    response read at ``output_xyz`` direction ``output_dir``.

    The output is in millimetres per second squared per newton, matching
    :class:`pymodal.frf` defaults.
    """
    if damping is None:
        damping = geom.damping
    freqs, V, _ = modes(geom)
    omega = 2.0 * np.pi * freq_array
    omega_n = 2.0 * np.pi * freqs
    b = point_to_dof_vector(input_xyz, input_dir, geom)
    c = point_to_dof_vector(output_xyz, output_dir, geom)
    bphi = V.T @ b
    cphi = V.T @ c
    H_disp = np.zeros_like(omega, dtype=complex)
    for r in range(len(freqs)):
        if omega_n[r] < 1e-3:
            continue
        num = cphi[r] * bphi[r]
        den = (omega_n[r] ** 2 - omega ** 2) + 2j * damping * omega_n[r] * omega
        H_disp += num / den
    H_acc_si = -omega ** 2 * H_disp          # m / s^2 / N
    return H_acc_si * 1.0e3                   # -> mm / s^2 / N


def compute_frf_matrix(freq_array: np.ndarray,
                        inputs: list,
                        outputs: list,
                        geom: BuildingGeometry,
                        damping: Optional[float] = None) -> np.ndarray:
    """Compute every (output, input) FRF for one geometry in a single pass.

    ``inputs`` and ``outputs`` are lists of ``(point, direction)`` tuples.
    Returns a ``(n_freq, n_outputs, n_inputs)`` complex array, ready to drop
    into :class:`pymodal.frf` as a single item.
    """
    if damping is None:
        damping = geom.damping
    freqs, V, _ = modes(geom)
    omega = 2.0 * np.pi * freq_array
    omega_n = 2.0 * np.pi * freqs

    B = np.stack([point_to_dof_vector(p, d, geom) for p, d in inputs], axis=1)   # (9, n_in)
    C = np.stack([point_to_dof_vector(p, d, geom) for p, d in outputs], axis=1)  # (9, n_out)
    bphi = V.T @ B          # (n_modes, n_in)
    cphi = V.T @ C          # (n_modes, n_out)

    valid = omega_n > 1e-3
    omega_n = omega_n[valid]
    bphi = bphi[valid]
    cphi = cphi[valid]

    # Sum over modes of c_r b_r / D_r(omega), broadcasted over frequencies
    den = (omega_n[:, None] ** 2 - omega[None, :] ** 2) + \
          2j * damping * omega_n[:, None] * omega[None, :]    # (n_modes, n_freq)
    # H_disp[i, j, k] = sum_r cphi[r, i] * bphi[r, j] / den[r, k]
    inv_den = 1.0 / den                                       # (n_modes, n_freq)
    H_disp = np.einsum('ri,rj,rk->kij', cphi, bphi, inv_den)
    H_acc = -omega[:, None, None] ** 2 * H_disp
    return H_acc * 1.0e3
