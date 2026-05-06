"""Reduced-order modal model of the Los Alamos 3-storey benchmark.

The full FEA pipeline in :mod:`salome_build` and :mod:`frf_extract` is the
high-fidelity workhorse, but generating thousands of FRFs through Code_Aster
is prohibitively slow for a notebook demo. This module provides a cheap,
physics-based alternative built around three modelling choices:

* every floor plate is a rigid body with three in-plane DOFs ``(x, y, θ)``
  at its centroid, except the base plate which is rolling on rails along
  ``RAIL_DIRECTION`` (its ``X`` and ``Z`` translations and all rotations
  are locked, so it keeps a single DOF - translation along the rails);
* columns are Euler-Bernoulli fixed-fixed beams contributing translational
  stiffness ``k = 12 E I / L^3`` at their footprint, with the column
  cross-section uniformly scaled by ``column_factor[storey, corner] ∈
  (0, 1]`` to encode progressive thinning damage. ``factor = 1`` is the
  pristine column, ``factor = 0`` removes the column entirely;
* every column terminates at the mid-thickness of the plates it joins
  (including base and roof), so the effective beam length is identical
  for every storey: ``L = inter_storey_gap + plate_lz``.

The DOF count is therefore ``1 + 3 * n_stories`` (10 for the LANL geometry)
and a single ``scipy.linalg.eigh`` solves the eigenproblem in microseconds.

The same ``(point, direction)`` call signature as :mod:`frf_extract` is
preserved, so anything written against :func:`compute_frf_matrix` here
keeps working when the modal provider is swapped for an FEA wrapper on any
other mesh.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

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

    Defaults pull from :mod:`params` so the unperturbed geometry matches
    the nominal LANL 3-storey benchmark. Each numeric field can be
    perturbed independently to manufacture small geometric variations.

    ``column_factor`` is a ``(n_stories, 4)`` array of section-scale
    factors. Both column dimensions are multiplied by the corresponding
    factor at every storey/corner, so the lateral stiffness contribution
    of that column scales as ``factor**4`` (because ``k = 12 E I / L^3``
    and ``I ∝ a^3 b ∝ factor^4`` under uniform shrink). ``factor = 1`` is
    pristine; ``factor → 0`` is total removal.
    """

    plate_lx: float = P.PLATE_LX
    plate_ly: float = P.PLATE_LY
    plate_lz: float = P.PLATE_LZ
    col_lx: float = P.COL_LX
    col_ly: float = P.COL_LY
    inter_storey_gap: float = P.INTER_STOREY_GAP
    column_gap: float = P.COLUMN_GAP
    n_stories: int = P.N_STORIES
    young: float = P.ALU_E
    poisson: float = P.ALU_NU
    density: float = P.ALU_RHO
    damping: float = P.MODAL_DAMPING
    rail_direction: str = P.RAIL_DIRECTION
    column_factor: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.column_factor is None:
            self.column_factor = np.ones((self.n_stories, 4), dtype=float)
        else:
            self.column_factor = np.asarray(self.column_factor, dtype=float)

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

    def column_box_centres(self) -> np.ndarray:
        """``(4, 2)`` array of the **column box geometric centres** in
        plate-local coordinates. Used for visualisation and for placing the
        column volumes in Salome.

        Columns sit externally to the plate footprint along ``Y``: two on the
        ``-Y`` side, two on the ``+Y`` side. Order:
        ``(low-x -Y, high-x -Y, low-x +Y, high-x +Y)``.
        """
        x_lo_centre = self.col_lx / 2.0
        x_hi_centre = self.plate_lx - self.col_lx / 2.0
        y_neg = -self.col_ly / 2.0 - self.column_gap
        y_pos = self.plate_ly + self.col_ly / 2.0 + self.column_gap
        return np.array([(x_lo_centre, y_neg),
                          (x_hi_centre, y_neg),
                          (x_lo_centre, y_pos),
                          (x_hi_centre, y_pos)])

    def column_attachment_points(self) -> np.ndarray:
        """``(4, 2)`` array of the **plate-side screw centroids** for each
        column, in plate-local coordinates.

        These are the effective points where each column transmits force to
        the plate (the centroid of its two screws on the plate's side face).
        Used by :func:`stiffness_matrix` for the kinematic transformation
        from plate DOFs to column-end displacements.

        Same ordering as :meth:`column_box_centres`.
        """
        x_lo_centre = self.col_lx / 2.0
        x_hi_centre = self.plate_lx - self.col_lx / 2.0
        return np.array([(x_lo_centre, 0.0),
                          (x_hi_centre, 0.0),
                          (x_lo_centre, self.plate_ly),
                          (x_hi_centre, self.plate_ly)])

    @property
    def n_dof(self) -> int:
        """1 (rolling base) + 3 per upper plate."""
        return 1 + 3 * self.n_stories

    def upper_dof_slice(self, plate_index: int) -> Tuple[int, int, int]:
        """Return the ``(x, y, theta)`` DOF indices of upper plate
        ``plate_index`` (1..n_stories). Base plate (index 0) only has DOF
        0, the rail translation, so callers must special-case it."""
        if not (1 <= plate_index <= self.n_stories):
            raise ValueError("plate_index must be in 1..n_stories for an upper plate")
        s = plate_index
        return 3 * s - 2, 3 * s - 1, 3 * s


# ---------------------------------------------------------------------------
# Stiffness and mass matrices
# ---------------------------------------------------------------------------
def _column_lateral_stiffnesses(geom: BuildingGeometry) -> Tuple[float, float]:
    """Translational stiffness of one *nominal* (factor = 1) fixed-fixed
    column in the X and Y directions (force per unit relative end
    displacement). Per-column scaling by ``factor**4`` is applied at
    assembly time."""
    L = geom.storey_height
    I_yy = geom.col_ly * geom.col_lx ** 3 / 12.0   # bending about y, deflection in x
    I_xx = geom.col_lx * geom.col_ly ** 3 / 12.0   # bending about x, deflection in y
    kx = 12.0 * geom.young * I_yy / L ** 3
    ky = 12.0 * geom.young * I_xx / L ** 3
    return kx, ky


def _column_local_K(kx_eff: float, ky_eff: float) -> np.ndarray:
    """4 x 4 local stiffness in (u_x_top, u_y_top, u_x_bot, u_y_bot)."""
    K = np.zeros((4, 4))
    K[0, 0] =  kx_eff; K[0, 2] = -kx_eff
    K[1, 1] =  ky_eff; K[1, 3] = -ky_eff
    K[2, 0] = -kx_eff; K[2, 2] =  kx_eff
    K[3, 1] = -ky_eff; K[3, 3] =  ky_eff
    return K


def _T_top(xc: float, yc: float) -> np.ndarray:
    """2 x 3 kinematic map ``(x_plate, y_plate, theta_plate) -> (u_x, u_y)``
    at column-centre offset ``(xc, yc)`` from the plate centroid."""
    return np.array([[1.0, 0.0, -yc],
                     [0.0, 1.0,  xc]])


def _T_base(xc: float, yc: float, geom: BuildingGeometry) -> np.ndarray:
    """2 x 1 kinematic map ``y_base -> (u_x_base, u_y_base)``.

    With rails along ``RAIL_DIRECTION`` the only free base DOF is the
    rail translation. The opposite direction is locked, so its row is
    zero. Rotations are also locked, so neither component depends on
    ``(xc, yc)`` - the base translates rigidly.
    """
    rail = geom.rail_direction.upper()
    T = np.zeros((2, 1))
    if rail == 'Y':
        T[1, 0] = 1.0          # u_y_base = y_base, u_x_base = 0
    elif rail == 'X':
        T[0, 0] = 1.0          # u_x_base = x_base, u_y_base = 0
    else:
        raise ValueError("RAIL_DIRECTION must be 'X' or 'Y'; got " + repr(rail))
    return T


def stiffness_matrix(geom: BuildingGeometry) -> np.ndarray:
    """Assemble the ``(1 + 3 n_stories)``-square lateral stiffness matrix.

    DOF ordering: ``[y_base, x_1, y_1, theta_1, x_2, y_2, theta_2, ...]``.
    """
    n = geom.n_stories
    n_dof = geom.n_dof
    K = np.zeros((n_dof, n_dof))
    cx, cy = geom.plate_centroid
    # Use the plate-side screw centroids as the kinematic attachment points;
    # this is where each column actually transmits force to the plate.
    centres = geom.column_attachment_points()
    kx, ky = _column_lateral_stiffnesses(geom)

    for s in range(n):                    # 0-indexed storey, 0 = bottom
        for c, (xc_abs, yc_abs) in enumerate(centres):
            factor = float(geom.column_factor[s, c])
            if factor <= 0.0:
                continue
            scale = factor ** 4
            xc = xc_abs - cx
            yc = yc_abs - cy
            K_local = _column_local_K(kx * scale, ky * scale)
            T_top = _T_top(xc, yc)        # (2, 3)
            if s == 0:
                T_bot = _T_base(xc, yc, geom)  # (2, 1)
                # 4 x 4 with column order [y_base, x_1, y_1, theta_1]:
                #   rows 0,1 = u_x_top, u_y_top, depend on top DOFs only
                #   rows 2,3 = u_x_bot, u_y_bot, depend on base DOF only
                T = np.zeros((4, 4))
                T[:2, 1:4] = T_top
                T[2:, 0:1] = T_bot
                idx = [0, 1, 2, 3]              # [y_base, x_1, y_1, theta_1]
            else:
                T_bot = T_top
                T = np.zeros((4, 6))
                T[:2, 0:3] = T_top              # top DOFs -> u_top
                T[2:, 3:6] = T_bot              # bot DOFs -> u_bot
                ix_top = list(geom.upper_dof_slice(s + 1))
                ix_bot = list(geom.upper_dof_slice(s))
                idx = ix_top + ix_bot
            K_block = T.T @ K_local @ T
            for i_loc, i_glob in enumerate(idx):
                for j_loc, j_glob in enumerate(idx):
                    K[i_glob, j_glob] += K_block[i_loc, j_loc]
    return K


def mass_matrix(geom: BuildingGeometry) -> np.ndarray:
    """Lumped mass / polar-inertia matrix matching :func:`stiffness_matrix`.

    The base plate's only retained DOF is the rail translation, so its
    contribution is a single diagonal entry equal to the plate mass.
    """
    n = geom.n_stories
    n_dof = geom.n_dof
    M = np.zeros((n_dof, n_dof))
    m = geom.density * geom.plate_lx * geom.plate_ly * geom.plate_lz
    J = m * (geom.plate_lx ** 2 + geom.plate_ly ** 2) / 12.0
    M[0, 0] = m
    for s in range(1, n + 1):
        ix, iy, it = geom.upper_dof_slice(s)
        M[ix, ix] = m
        M[iy, iy] = m
        M[it, it] = J
    return M


def modes(geom: BuildingGeometry):
    """Solve the generalised eigenvalue problem ``K v = w**2 M v``.

    Returns ``(freqs_hz, V, M)`` with ``V`` mass-normalised
    (``V.T @ M @ V == I``) and ``freqs_hz`` ascending.
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
    """Map a point/direction on the building to its participation vector.

    Plate identification is by closest mid-thickness ``z``. For the base
    plate (index 0) only the rail-direction DOF is excited. For upper
    plates the rigid-body kinematics ``u = u_centroid + ω × r`` apply.
    Out-of-plane (Z) excitations are not represented in this in-plane
    model and return a zero vector.
    """
    direction = direction.upper().strip()
    if direction not in ('X', 'Y', 'Z'):
        raise ValueError("direction must be 'X', 'Y' or 'Z'")
    x, y, z = point
    cx, cy = geom.plate_centroid
    rx = x - cx
    ry = y - cy

    plate_centres = np.array([geom.plate_z_centre(k)
                               for k in range(geom.n_stories + 1)])
    plate_idx = int(np.argmin(np.abs(plate_centres - z)))

    b = np.zeros(geom.n_dof)
    if plate_idx == 0:
        # base plate: only the rail-direction DOF is excited
        if direction == geom.rail_direction.upper():
            b[0] = 1.0
        return b

    s = plate_idx                             # 1..n_stories
    ix, iy, it = geom.upper_dof_slice(s)
    if direction == 'X':
        b[ix] = 1.0
        b[it] = -ry
    elif direction == 'Y':
        b[iy] = 1.0
        b[it] = rx
    # Z: no DOFs in this in-plane model
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
    """Accelerance ``H_acc(omega) = -omega^2 * H_disp(omega)`` for a unit
    harmonic force at ``input_xyz`` direction ``input_dir`` and a response
    read at ``output_xyz`` direction ``output_dir``. Output in
    ``mm / s^2 / N`` (pymodal default)."""
    if damping is None:
        damping = geom.damping
    freqs, V, _ = modes(geom)
    omega = 2.0 * np.pi * freq_array
    omega_n = 2.0 * np.pi * freqs
    b = point_to_dof_vector(input_xyz, input_dir, geom)
    c = point_to_dof_vector(output_xyz, output_dir, geom)
    bphi = V.T @ b
    cphi = V.T @ c

    # Build accelerance directly. The rail rigid-body mode (omega_n = 0)
    # contributes a frequency-independent inertial term equal to
    # cphi * bphi (mass-normalised modes); for elastic modes the standard
    # modal-superposition kernel applies, multiplied by -omega^2 to turn
    # displacement into acceleration.
    H_acc = np.zeros_like(omega, dtype=complex)
    for r in range(len(freqs)):
        num = cphi[r] * bphi[r]
        if omega_n[r] < 1e-3:
            H_acc += num
        else:
            den = (omega_n[r] ** 2 - omega ** 2) \
                  + 2j * damping * omega_n[r] * omega
            H_acc += -omega ** 2 * num / den
    return H_acc * 1.0e3                       # m/s^2/N -> mm/s^2/N


def compute_frf_matrix(freq_array: np.ndarray,
                        inputs: Sequence,
                        outputs: Sequence,
                        geom: BuildingGeometry,
                        damping: Optional[float] = None) -> np.ndarray:
    """Vectorised version: every ``(output, input)`` FRF in one pass.

    ``inputs``/``outputs`` are sequences of ``(point, direction)`` tuples.
    Returns a ``(n_freq, n_outputs, n_inputs)`` complex array, ready as a
    single :class:`pymodal.frf` item.
    """
    if damping is None:
        damping = geom.damping
    freqs, V, _ = modes(geom)
    omega = 2.0 * np.pi * freq_array
    omega_n = 2.0 * np.pi * freqs

    B = np.stack([point_to_dof_vector(p, d, geom) for p, d in inputs], axis=1)
    C = np.stack([point_to_dof_vector(p, d, geom) for p, d in outputs], axis=1)
    bphi = V.T @ B          # (n_modes, n_in)
    cphi = V.T @ C          # (n_modes, n_out)

    is_rigid = omega_n < 1e-3
    is_elastic = ~is_rigid

    # Elastic-mode contribution
    elastic_H_acc = np.zeros((len(omega), C.shape[1], B.shape[1]), dtype=complex)
    if is_elastic.any():
        wn = omega_n[is_elastic]
        b_e = bphi[is_elastic]
        c_e = cphi[is_elastic]
        den = (wn[:, None] ** 2 - omega[None, :] ** 2) + \
              2j * damping * wn[:, None] * omega[None, :]
        kernel = -omega[None, :] ** 2 / den                  # (n_modes, n_freq)
        elastic_H_acc = np.einsum('ri,rj,rk->kij', c_e, b_e, kernel)

    # Rigid-body (rail) mode contribution: a frequency-independent
    # +cphi*bphi added to every frequency bin
    rigid_H_acc = np.zeros((C.shape[1], B.shape[1]), dtype=complex)
    if is_rigid.any():
        rigid_H_acc = np.einsum('ri,rj->ij', cphi[is_rigid], bphi[is_rigid])

    H_acc = elastic_H_acc + rigid_H_acc[None, :, :]
    return H_acc * 1.0e3
