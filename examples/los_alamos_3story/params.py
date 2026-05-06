"""Geometry, material, and analysis parameters for the Los Alamos 3-storey
building benchmark.

All lengths are in metres, masses in kilograms, times in seconds, frequencies
in hertz. Default values follow the published dimensions of the LANL 3-storey
benchmark structure (Figueiredo et al., LA-14393), but every quantity is just
a module-level attribute, so the user is free to override any of them before
calling :mod:`salome_build` or :mod:`frf_extract`.

Layout convention
-----------------
The structure has ``N_STORIES`` stories and therefore ``N_STORIES + 1`` floor
plates, indexed from ``0`` (base plate, fixed to ground) to ``N_STORIES``
(roof plate). Four columns rise at the corners of every storey. Following the
project specification:

* every column joins each adjacent plate by reaching half the plate's
  thickness, so two columns meet at the mid-thickness of each intermediate
  plate;
* the columns at the base storey reach through the full thickness of the
  base plate (their bottom face sits on the ground);
* the columns at the top storey reach through the full thickness of the roof
  plate (their top face is flush with the roof).

Each column-plate junction is bolted with one screw (a small cylinder) running
parallel to the column axis through the column section and the adjacent plate.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Plates (floors)
# ---------------------------------------------------------------------------
PLATE_LX: float = 0.305      # m, plate side along x
PLATE_LY: float = 0.305      # m, plate side along y
PLATE_LZ: float = 0.0254     # m, plate thickness (vertical)

# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
COL_LX: float = 0.0254       # m, column cross-section along x
COL_LY: float = 0.0064       # m, column cross-section along y (thin direction)
INTER_STOREY_GAP: float = 0.1524   # m, vertical clearance between two plates
COLUMN_INSET: float = 0.0127       # m, column outer-face offset from plate edge

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
N_STORIES: int = 3           # 3-storey building => 4 plates total

# ---------------------------------------------------------------------------
# Screws (two per column-plate junction, offset along the column wide axis)
# ---------------------------------------------------------------------------
SCREW_DIAMETER: float = 0.005   # m, screw shank diameter
SCREW_OFFSET_FRAC: float = 0.30 # screw offset from column centre, as a
                                # fraction of (col_lx / 2). Two screws sit
                                # symmetrically at +/- this offset along x.

# ---------------------------------------------------------------------------
# Base rails: the bottom face of the base plate is constrained so that only
# translation along ``RAIL_DIRECTION`` is free (rolls on rails). This makes
# the building move along the columns' thin side - the soft direction.
# ---------------------------------------------------------------------------
RAIL_DIRECTION: str = 'Y'       # the only free DOF on the base; X and Z fixed

# ---------------------------------------------------------------------------
# Material (6061-T6 aluminium)
# ---------------------------------------------------------------------------
ALU_E: float = 70.0e9        # Pa, Young's modulus
ALU_NU: float = 0.33         # Poisson's ratio
ALU_RHO: float = 2700.0      # kg/m^3
MODAL_DAMPING: float = 0.005 # uniform modal damping ratio (0.5 %)

# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
MESH_SIZE: float = 0.012     # m, target tetrahedral edge length

# ---------------------------------------------------------------------------
# Default analysis settings
# ---------------------------------------------------------------------------
F_MIN: float = 0.0           # Hz, lower bound of FRF frequency axis
F_MAX: float = 200.0         # Hz, upper bound
F_STEP: float = 0.5          # Hz, frequency resolution
N_MODES: int = 60            # number of modes used for modal projection


# ---------------------------------------------------------------------------
# Derived helpers (kept as functions to remain reactive to overrides)
# ---------------------------------------------------------------------------
def storey_height() -> float:
    """Vertical pitch between two consecutive plate bottoms."""
    return PLATE_LZ + INTER_STOREY_GAP


def plate_z_bottom(k: int) -> float:
    """Z-coordinate of the bottom face of plate index ``k`` (k = 0..N_STORIES)."""
    return k * storey_height()


def total_height() -> float:
    """Total height of the building from the ground to the top of the roof."""
    return plate_z_bottom(N_STORIES) + PLATE_LZ


def column_positions():
    """``(x, y)`` of the lower-x / lower-y corner of each of the four columns,
    common to every storey."""
    return (
        (COLUMN_INSET, COLUMN_INSET),
        (PLATE_LX - COLUMN_INSET - COL_LX, COLUMN_INSET),
        (COLUMN_INSET, PLATE_LY - COLUMN_INSET - COL_LY),
        (PLATE_LX - COLUMN_INSET - COL_LX, PLATE_LY - COLUMN_INSET - COL_LY),
    )


def column_z_extent(storey_index: int):
    """``(z_start, z_end)`` of the column belonging to storey ``storey_index``
    (0-based, so ``0`` is the bottom storey, ``N_STORIES - 1`` the top).

    Every column - including those at the bottom and top storey - terminates
    at the **mid-thickness** of the plate it joins, so column lengths are
    identical for every storey.
    """
    z_start = plate_z_bottom(storey_index)     + PLATE_LZ / 2.0
    z_end   = plate_z_bottom(storey_index + 1) + PLATE_LZ / 2.0
    return z_start, z_end
