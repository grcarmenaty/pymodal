"""Build the Los Alamos 3-storey benchmark in Salome and export a MED mesh.

This script is meant to be executed inside Salome's Python interpreter, e.g.::

    salome -t examples/los_alamos_3story/salome_build.py

It produces three files next to itself:

* ``building.med``   - the volumetric tetrahedral mesh
* ``nodes.json``     - mapping ``node_id -> [x, y, z]`` for every mesh node,
                       used by :mod:`frf_extract` to resolve user-supplied
                       coordinates to the closest mesh node
* ``groups.json``    - the names of the boundary-condition node/face groups
                       written into the MED file (handy when editing the
                       Code_Aster command template)

Geometry layout follows :mod:`params`:

* every plate is a parallelepiped of dimensions
  ``(PLATE_LX, PLATE_LY, PLATE_LZ)``, stacked along ``z`` with an inter-storey
  clearance of ``INTER_STOREY_GAP``;
* every column is a parallelepiped of cross-section ``(COL_LX, COL_LY)``
  rising along ``z`` and terminating at the **mid-thickness** of the two
  plates it joins, sitting **externally** to the plate footprint along the
  ``Y`` direction with a small gap ``COLUMN_GAP``;
* every column-plate junction is bolted with **two** cylindrical screws of
  diameter ``SCREW_DIAMETER`` drilling **perpendicular** to the column's
  biggest face (``COL_LX × column_height``), i.e. along the ``Y`` axis,
  through the column thickness, across the gap, and into the plate's side
  face by ``SCREW_PLATE_DEPTH``;
* the column and plate volumes never touch (the gap separates them); the
  only solid bridge is the screws, which are fused to both columns and
  plates;
* the bottom face of the base plate is exported as the ``base_rails`` face
  group so the Code_Aster command file can lock all DOFs except translation
  along ``RAIL_DIRECTION``.
"""
import json
import os
import sys

# Make ``params`` importable regardless of the directory salome is launched from
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import params as P  # noqa: E402

import salome  # noqa: E402

salome.salome_init()

import GEOM  # noqa: E402,F401
from salome.geom import geomBuilder  # noqa: E402
import SMESH  # noqa: E402
from salome.smesh import smeshBuilder  # noqa: E402

geompy = geomBuilder.New()
smesh = smeshBuilder.New()


# ---------------------------------------------------------------------------
# 1. Plates
# ---------------------------------------------------------------------------
plates = []
for k in range(P.N_STORIES + 1):
    plate = geompy.MakeBoxDXDYDZ(P.PLATE_LX, P.PLATE_LY, P.PLATE_LZ)
    plate = geompy.MakeTranslation(plate, 0.0, 0.0, P.plate_z_bottom(k))
    geompy.addToStudy(plate, "plate_%d" % k)
    plates.append(plate)


# ---------------------------------------------------------------------------
# 2. Columns (external; one column-segment per storey, per corner)
# ---------------------------------------------------------------------------
columns = []
for storey in range(P.N_STORIES):
    z_start, z_end = P.column_z_extent(storey)
    col_height = z_end - z_start
    for c, (x0, y0) in enumerate(P.column_positions()):
        col = geompy.MakeBoxDXDYDZ(P.COL_LX, P.COL_LY, col_height)
        col = geompy.MakeTranslation(col, x0, y0, z_start)
        geompy.addToStudy(col, "column_s%d_c%d" % (storey, c))
        columns.append(col)


# ---------------------------------------------------------------------------
# 3. Screws (two per column-plate junction; drill along +/- Y, perpendicular
#    to the column's biggest face)
# ---------------------------------------------------------------------------
def _screw_along_y(x_screw, y_start, length, z_screw, diameter):
    """Cylinder along +Y starting at ``(x_screw, y_start, z_screw)`` with
    the requested ``length``."""
    base = geompy.MakeVertex(x_screw, y_start, z_screw)
    axis = geompy.MakeVectorDXDYDZ(0.0, 1.0, 0.0)
    return geompy.MakeCylinder(base, axis, diameter / 2.0, length)


SCREW_TOTAL_LENGTH = P.COL_LY + P.COLUMN_GAP + P.SCREW_PLATE_DEPTH
SCREW_X_OFFSET = P.SCREW_OFFSET_FRAC * P.COL_LX / 2.0
SCREW_Z_OFFSET = P.SCREW_Z_OFFSET_FRAC * P.PLATE_LZ

screws = []
positions = P.column_positions()
for storey in range(P.N_STORIES):
    z_low_centre, z_high_centre = P.column_z_extent(storey)
    for c, (x0, y0) in enumerate(positions):
        col_x_centre = x0 + P.COL_LX / 2.0
        # Determine which side the column is on, hence the screw's y_start
        # (the cylinder always grows in +Y; for +Y-side columns we shift
        # the starting y so the cylinder's tail ends up inside the plate)
        if y0 < 0.0:
            # column on -Y side: outer face at y0; screw enters at y0,
            # exits at y0 + SCREW_TOTAL_LENGTH = +SCREW_PLATE_DEPTH
            y_start = y0
        else:
            # column on +Y side: inner end of the screw is at PLATE_LY -
            # SCREW_PLATE_DEPTH (inside plate); outer end at y_start + length
            y_start = P.PLATE_LY - P.SCREW_PLATE_DEPTH

        # Bottom screws: inside the lower plate, just above its mid-thickness
        z_bottom = z_low_centre + SCREW_Z_OFFSET
        # Top screws: inside the upper plate, just below its mid-thickness
        z_top = z_high_centre - SCREW_Z_OFFSET

        for x_off in (-SCREW_X_OFFSET, +SCREW_X_OFFSET):
            x_screw = col_x_centre + x_off
            screws.append(_screw_along_y(
                x_screw, y_start, SCREW_TOTAL_LENGTH, z_bottom,
                P.SCREW_DIAMETER))
            screws.append(_screw_along_y(
                x_screw, y_start, SCREW_TOTAL_LENGTH, z_top,
                P.SCREW_DIAMETER))

for i, s in enumerate(screws):
    geompy.addToStudy(s, "screw_%03d" % i)


# ---------------------------------------------------------------------------
# 4. Fuse everything into one connected solid
# ---------------------------------------------------------------------------
# Plates and columns do NOT overlap or share faces (separated by COLUMN_GAP),
# so the fuse only welds along the volumes that DO overlap: each screw
# overlaps with the column it pierces and with the plate it threads into.
# The result is a single connected solid where every column is bonded to
# every adjacent plate exclusively through the screw cylinders.
building = geompy.MakeFuseList(plates + columns + screws,
                                checkSelfInte=False, rmExtraEdges=True)
geompy.addToStudy(building, "los_alamos_building")


# ---------------------------------------------------------------------------
# 5. Boundary-condition group: bottom face of base plate (the rails)
# ---------------------------------------------------------------------------
base_centre = geompy.MakeVertex(P.PLATE_LX / 2.0, P.PLATE_LY / 2.0, 0.0)
base_face = geompy.GetFaceNearPoint(building, base_centre)
base_group = geompy.CreateGroup(building, geompy.ShapeType["FACE"])
geompy.UnionList(base_group, [base_face])
geompy.addToStudyInFather(building, base_group, "base_rails")


# ---------------------------------------------------------------------------
# 6. Mesh with NETGEN 1D-2D-3D
# ---------------------------------------------------------------------------
mesh = smesh.Mesh(building, "los_alamos_mesh")
algo = mesh.Tetrahedron(algo=smeshBuilder.NETGEN_1D2D3D)
algo_params = algo.Parameters()
algo_params.SetMaxSize(P.MESH_SIZE)
algo_params.SetMinSize(P.MESH_SIZE / 5.0)
algo_params.SetSecondOrder(0)
algo_params.SetOptimize(1)
algo_params.SetFineness(2)        # moderate
algo_params.SetQuadAllowed(0)

ok = mesh.Compute()
if not ok:
    raise RuntimeError("NETGEN failed to mesh the fused building geometry")

mesh.GroupOnGeom(base_group, "base_rails", SMESH.FACE)


# ---------------------------------------------------------------------------
# 7. Export MED + companion JSON files
# ---------------------------------------------------------------------------
mesh_path = os.path.join(_HERE, "building.med")
mesh.ExportMED(mesh_path, auto_groups=False, version=41)
print("Mesh written to %s (%d nodes, %d tetrahedra)"
      % (mesh_path, mesh.NbNodes(), mesh.NbTetras()))

node_ids = mesh.GetNodesId()
nodes = {int(nid): list(mesh.GetNodeXYZ(nid)) for nid in node_ids}
with open(os.path.join(_HERE, "nodes.json"), "w") as fp:
    json.dump(nodes, fp)

groups = {"rails_face_group": "base_rails", "rail_direction": P.RAIL_DIRECTION}
with open(os.path.join(_HERE, "groups.json"), "w") as fp:
    json.dump(groups, fp, indent=2)

print("Saved %d node coordinates to nodes.json" % len(nodes))
print("Group names dumped to groups.json: %s" % groups)
