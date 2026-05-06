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
(roof plate). Four columns rise at the **outside** of the four corners of
every storey, with their thin direction (``COL_LY``) aligned with the
rail/motion axis ``Y``. Column-plate connections follow the project
specification:

* every column terminates at the **mid-thickness** of the two plates it
  joins (z-axis), so the column's z-extent is from ``mid-plate(s)`` to
  ``mid-plate(s+1)`` for storey ``s``;
* the columns sit **externally** to the plate footprint along the ``Y``
  direction with a small gap ``COLUMN_GAP``: two columns on the ``-Y`` side
  (low-y corners) and two on the ``+Y`` side (high-y corners);
* the column's biggest face is the ``COL_LX × column_height`` face (since
  ``COL_LX > COL_LY``); each column-plate junction is bolted with **two**
  screws drilling **perpendicular** to that biggest face (along the ``Y``
  axis), passing through the column thickness, across the gap, and into
  the **side** of the floor plate by ``SCREW_PLATE_DEPTH``;
* the column and plate volumes do NOT touch (the gap separates them) - the
  *only* solid bridge is each pair of screws, which is solid-fused to both
  the column and the plate.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Plates (floors)
# ---------------------------------------------------------------------------
PLATE_LX: float = 0.305      # m, plate side along x
PLATE_LY: float = 0.305      # m, plate side along y
PLATE_LZ: float = 0.0254     # m, plate thickness (vertical)

# ---------------------------------------------------------------------------
# Columns (external to plates, screwed to plate sides)
# ---------------------------------------------------------------------------
COL_LX: float = 0.0254       # m, column wide cross-section dimension (along x)
COL_LY: float = 0.0064       # m, column thin cross-section dimension (along y)
INTER_STOREY_GAP: float = 0.1524    # m, vertical clearance between two plates
COLUMN_GAP: float = 0.0005   # m, horizontal (y) gap between column inner face
                             # and plate side face. Small gap so column and
                             # plate volumes never touch directly; the only
                             # solid bond is via the screws.

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
N_STORIES: int = 3           # 3-storey building => 4 plates total

# ---------------------------------------------------------------------------
# Screws (two per column-plate junction; drill perpendicular to the column's
# biggest face - i.e. along the Y axis - passing through the column's wide
# face and into the plate's side face. Screws are spaced symmetrically
# along the column's wide cross-section axis X)
# ---------------------------------------------------------------------------
SCREW_DIAMETER: float = 0.005     # m, screw shank diameter
SCREW_OFFSET_FRAC: float = 0.30   # screw offset from column centre, as a
                                  # fraction of (COL_LX / 2). Two screws sit
                                  # symmetrically at +/- this offset along X.
SCREW_PLATE_DEPTH: float = 0.012  # m, depth of penetration into the plate
                                  # side. Total screw length is
                                  # COL_LY + COLUMN_GAP + SCREW_PLATE_DEPTH.
SCREW_Z_OFFSET_FRAC: float = 0.25 # vertical offset of the screw centre from
                                  # the plate's mid-thickness, as a fraction
                                  # of PLATE_LZ. The lower segment's top
                                  # screws are placed at
                                  # plate_z_centre - SCREW_Z_OFFSET_FRAC*PLATE_LZ
                                  # and the upper segment's bottom screws at
                                  # plate_z_centre + SCREW_Z_OFFSET_FRAC*PLATE_LZ
                                  # so the two pairs do not coincide.

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
# Derived helpers (functions so they remain reactive to overrides)
# ---------------------------------------------------------------------------
def storey_height() -> float:
    """Vertical pitch between two consecutive plate bottoms."""
    return PLATE_LZ + INTER_STOREY_GAP


def plate_z_bottom(k: int) -> float:
    """Z-coordinate of the bottom face of plate index ``k`` (k = 0..N_STORIES)."""
    return k * storey_height()


def plate_z_centre(k: int) -> float:
    """Z-coordinate of the mid-thickness of plate index ``k``."""
    return plate_z_bottom(k) + PLATE_LZ / 2.0


def total_height() -> float:
    """Total height of the building from the ground to the top of the roof."""
    return plate_z_bottom(N_STORIES) + PLATE_LZ


def column_positions():
    """Return the ``(x0, y0)`` lower corner of each column box, in
    plate-local coordinates. Columns sit **externally** to the plate
    footprint along ``Y``.

    Order: ``(low-x -Y, high-x -Y, low-x +Y, high-x +Y)`` to match
    :func:`reduced_model.BuildingGeometry.column_box_centres`.
    """
    x_low  = 0.0
    x_high = PLATE_LX - COL_LX
    y_neg  = -COL_LY - COLUMN_GAP                # column on -Y side: outer face at y_neg
    y_pos  = PLATE_LY + COLUMN_GAP               # column on +Y side: inner face at y_pos
    return (
        (x_low,  y_neg),
        (x_high, y_neg),
        (x_low,  y_pos),
        (x_high, y_pos),
    )


def column_z_extent(storey_index: int):
    """``(z_start, z_end)`` of the column belonging to storey ``storey_index``.

    Every column terminates at the **mid-thickness** of the two plates it
    joins, so column lengths are identical for every storey.
    """
    z_start = plate_z_centre(storey_index)
    z_end   = plate_z_centre(storey_index + 1)
    return z_start, z_end
