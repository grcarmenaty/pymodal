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
  rising along ``z`` and terminates at the **mid-thickness** of every plate
  it joins (including the base and roof), so two columns meet inside every
  intermediate plate and base / top columns end at mid base / roof plate;
* every column-plate junction is bolted with **two** cylindrical screws of
  diameter ``SCREW_DIAMETER`` aligned with the column axis, offset
  symmetrically along the column wide direction (x) by
  ``SCREW_OFFSET_FRAC * COL_LX / 2``;
* the bottom face of the base plate is exported as the ``base_rails`` face
  group so the Code_Aster command file can lock all DOFs except translation
  along ``RAIL_DIRECTION``.

The columns, plates, and screws are fused into a single solid, which is then
meshed with NETGEN 1D-2D-3D using a target edge length of ``MESH_SIZE``.
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
# 2. Columns
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
# 3. Screws (two cylinders per column-plate junction, offset along x)
# ---------------------------------------------------------------------------
def _make_screw(cx, cy, z_centre, length, diameter):
    """Cylinder along +z centred at (cx, cy, z_centre)."""
    base = geompy.MakeVertex(cx, cy, z_centre - length / 2.0)
    axis = geompy.MakeVectorDXDYDZ(0.0, 0.0, 1.0)
    return geompy.MakeCylinder(base, axis, diameter / 2.0, length)


def _junction_screws(cx, cy, z_centre):
    """Two screws straddling the column centre along x."""
    dx = P.SCREW_OFFSET_FRAC * P.COL_LX / 2.0
    return [_make_screw(cx - dx, cy, z_centre,
                          length=2.0 * P.PLATE_LZ, diameter=P.SCREW_DIAMETER),
            _make_screw(cx + dx, cy, z_centre,
                          length=2.0 * P.PLATE_LZ, diameter=P.SCREW_DIAMETER)]


screws = []
# Every plate (base + N_STORIES floor plates) is bolted to every corner column
# that meets it. Each junction = 2 screws.
for plate_idx in range(P.N_STORIES + 1):
    z_centre = P.plate_z_bottom(plate_idx) + P.PLATE_LZ / 2.0
    for x0, y0 in P.column_positions():
        cx = x0 + P.COL_LX / 2.0
        cy = y0 + P.COL_LY / 2.0
        screws.extend(_junction_screws(cx, cy, z_centre))

for i, s in enumerate(screws):
    geompy.addToStudy(s, "screw_%03d" % i)


# ---------------------------------------------------------------------------
# 4. Fuse everything into one connected solid
# ---------------------------------------------------------------------------
building = geompy.MakeFuseList(plates + columns + screws,
                                checkSelfInte=False, rmExtraEdges=True)
geompy.addToStudy(building, "los_alamos_building")


# ---------------------------------------------------------------------------
# 5. Boundary-condition group: bottom face of base plate (the rails)
# ---------------------------------------------------------------------------
# The bottom face becomes the ``base_rails`` group. Code_Aster locks every
# DOF on that face except translation along ``params.RAIL_DIRECTION`` so the
# structure rolls on rails along the columns' thin side.
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
