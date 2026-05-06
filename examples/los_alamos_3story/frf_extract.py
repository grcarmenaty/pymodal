"""Extract a frequency response function from the Los Alamos 3-storey
benchmark, given an input excitation point and an output measurement point.

This module is the user-facing entry point of the example. It does **not**
require Salome to be running: it talks to a pre-built ``building.med`` mesh
through a Code_Aster command file.

Workflow
--------
1.  Build the mesh once::

        salome -t examples/los_alamos_3story/salome_build.py

2.  Compute as many FRFs as you want::

        from examples.los_alamos_3story.frf_extract import extract_frf

        H = extract_frf(input_xyz=(0.0,    0.0,    0.0254 + 0.1524),
                         input_dir='X',
                         output_xyz=(0.305, 0.305, 0.0254 + 0.1524),
                         output_dir='X')

    ``H`` is a :class:`pymodal.frf` collection with a single 1-item entry,
    which can immediately be appended to a larger collection, plotted, or fed
    to ``HDF5Dataset``.

The same call accepts lists of points to build a multi-item ``frf`` collection
in one shot - useful for sweeping the grid of measurement points required by
SHM studies.

CLI
---
For shell-driven workflows the module also exposes a ``__main__`` entry point::

    python -m examples.los_alamos_3story.frf_extract \
        --input  0.0 0.0 0.1778 \
        --output 0.305 0.305 0.1778 \
        --input-dir X --output-dir X \
        --out frf.h5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# Make ``pymodal`` and the local ``params`` module importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for path in (_REPO, _HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import params as P  # noqa: E402
from pymodal import frf  # noqa: E402

Point = Tuple[float, float, float]
Direction = str  # 'X', 'Y' or 'Z'


# ---------------------------------------------------------------------------
# Mesh-side helpers
# ---------------------------------------------------------------------------
def _load_nodes(nodes_json: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(node_ids, coords)`` arrays read from ``nodes.json``."""
    with open(nodes_json) as fp:
        raw = json.load(fp)
    node_ids = np.array(sorted(int(k) for k in raw), dtype=np.int64)
    coords = np.array([raw[str(int(n))] for n in node_ids], dtype=float)
    return node_ids, coords


def closest_node(point: Point,
                  node_ids: np.ndarray,
                  coords: np.ndarray) -> Tuple[int, Point]:
    """Return the mesh-node id closest to ``point`` and its actual coordinates."""
    d2 = np.sum((coords - np.asarray(point, dtype=float)) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return int(node_ids[idx]), tuple(coords[idx].tolist())


def _validate_dir(d: Direction) -> str:
    d = d.upper().strip()
    if d not in ('X', 'Y', 'Z'):
        raise ValueError("direction must be 'X', 'Y' or 'Z' (got %r)" % d)
    return 'D' + d


# ---------------------------------------------------------------------------
# Code_Aster job orchestration
# ---------------------------------------------------------------------------
def _render_template(template_path: str, **subs) -> str:
    with open(template_path) as fp:
        return fp.read().format(**subs)


def _write_export(work: str,
                   comm: str,
                   med_in: str,
                   med_out: str,
                   resu: str,
                   mess: str,
                   table_unit_8: str,
                   memory_mb: int = 4096,
                   time_limit_s: int = 1800) -> str:
    """Generate a minimal Code_Aster ``.export`` file."""
    export_path = os.path.join(work, "job.export")
    with open(export_path, "w") as fp:
        fp.write("P actions make_etude\n")
        fp.write("P version stable\n")
        fp.write("P mode batch\n")
        fp.write("P time_limit %d\n" % time_limit_s)
        fp.write("A memjeveux %d\n" % memory_mb)
        fp.write("A memjob %d\n"   % memory_mb)
        fp.write("F comm %s D 1\n"  % comm)
        fp.write("F mmed %s D 20\n" % med_in)
        fp.write("F mess %s R 6\n"  % mess)
        fp.write("F resu %s R 8\n"  % table_unit_8)
        fp.write("F med  %s R 80\n" % med_out)
    return export_path


def _run_aster(export_path: str) -> None:
    aster_runner = os.environ.get("ASTER_RUNNER", "as_run")
    cmd = [aster_runner, export_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Code_Aster job failed (exit %d):\nstdout:\n%s\nstderr:\n%s"
            % (proc.returncode, proc.stdout, proc.stderr)
        )


def _parse_table(path: str, output_dir: str) -> np.ndarray:
    """Parse the IMPR_TABLE output and return a complex 1-D array of length n_freq.

    Code_Aster writes one line per ``(frequency, node)`` pair. Columns we
    expect: ``INTITULE`` (string), ``NOEUD`` (string), ``FREQ`` (real), and
    the requested DOF as a complex number with format ``(re,im)``.
    """
    with open(path) as fp:
        lines = [ln.strip() for ln in fp.readlines() if ln.strip()]
    # Find header row (first row containing 'FREQ')
    header_idx = next(i for i, ln in enumerate(lines) if 'FREQ' in ln.upper())
    header = [c.strip().upper() for c in lines[header_idx].split(',')]
    freq_col = header.index('FREQ')
    val_col = header.index(output_dir)
    freqs, vals = [], []
    for ln in lines[header_idx + 1:]:
        cells = [c.strip() for c in ln.split(',')]
        if len(cells) <= max(freq_col, val_col):
            continue
        try:
            freqs.append(float(cells[freq_col]))
            v = cells[val_col]
            if v.startswith('(') and v.endswith(')'):
                re_s, im_s = v[1:-1].split(',')
                vals.append(complex(float(re_s), float(im_s)))
            else:
                vals.append(complex(float(v), 0.0))
        except ValueError:
            continue
    return np.asarray(freqs, dtype=float), np.asarray(vals, dtype=complex)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_frf(
    input_xyz: Union[Point, Sequence[Point]],
    input_dir: Union[Direction, Sequence[Direction]] = 'X',
    output_xyz: Optional[Union[Point, Sequence[Point]]] = None,
    output_dir: Union[Direction, Sequence[Direction]] = 'X',
    *,
    f_min: float = P.F_MIN,
    f_max: float = P.F_MAX,
    f_step: float = P.F_STEP,
    n_modes: int = P.N_MODES,
    damping: float = P.MODAL_DAMPING,
    young: float = P.ALU_E,
    poisson: float = P.ALU_NU,
    density: float = P.ALU_RHO,
    mesh_dir: str = _HERE,
    template_path: Optional[str] = None,
    work_dir: Optional[str] = None,
    keep_work: bool = False,
    names: Optional[List[str]] = None,
    path: Optional[str] = None,
) -> frf:
    """Compute the FRF(s) from one or more input nodes to one or more output
    nodes and wrap the result in a :class:`pymodal.frf` collection.

    ``input_xyz`` and ``output_xyz`` accept either a single ``(x, y, z)``
    tuple or a list of tuples; lists must have the same length and produce a
    multi-item collection with one item per (input, output) pair.  When
    ``output_xyz`` is ``None`` it defaults to the corresponding entry of
    ``input_xyz`` (driving-point FRFs).
    """
    template_path = template_path or os.path.join(_HERE, "aster_frf_template.comm")
    nodes_json = os.path.join(mesh_dir, "nodes.json")
    groups_json = os.path.join(mesh_dir, "groups.json")
    med_path = os.path.join(mesh_dir, "building.med")
    for required in (nodes_json, groups_json, med_path):
        if not os.path.exists(required):
            raise FileNotFoundError(
                "Missing %s - did you run salome_build.py first?" % required
            )

    node_ids, coords = _load_nodes(nodes_json)
    with open(groups_json) as fp:
        groups = json.load(fp)
    rails_group = groups["rails_face_group"]
    rail_dir = "D" + groups.get("rail_direction", P.RAIL_DIRECTION).upper()
    locked = [d for d in ("DX", "DY", "DZ") if d != rail_dir]
    locked_dofs = ", ".join(f"{d}=0.0" for d in locked)

    # Normalise inputs to lists of equal length
    def _as_list(x):
        if x is None:
            return None
        if isinstance(x, (tuple, list)) and len(x) and isinstance(x[0], (int, float)):
            return [tuple(x)]
        return [tuple(p) for p in x]

    inputs = _as_list(input_xyz)
    outputs = _as_list(output_xyz) if output_xyz is not None else list(inputs)

    if isinstance(input_dir, str):
        input_dirs = [input_dir] * len(inputs)
    else:
        input_dirs = list(input_dir)
    if isinstance(output_dir, str):
        output_dirs = [output_dir] * len(outputs)
    else:
        output_dirs = list(output_dir)

    if not (len(inputs) == len(outputs) == len(input_dirs) == len(output_dirs)):
        raise ValueError(
            "input_xyz, output_xyz, input_dir and output_dir must agree in length"
        )

    work_root = work_dir or tempfile.mkdtemp(prefix="los_alamos_frf_")
    measurements: List[np.ndarray] = []
    coordinates: List[Tuple[Point, Point]] = []
    orientations: List[Tuple[str, str]] = []
    freq_array: Optional[np.ndarray] = None

    for k, (pi, po, di, do) in enumerate(zip(inputs, outputs, input_dirs, output_dirs)):
        in_node, in_coord = closest_node(pi, node_ids, coords)
        out_node, out_coord = closest_node(po, node_ids, coords)
        in_dof = _validate_dir(di)
        out_dof = _validate_dir(do)

        case_dir = os.path.join(work_root, "case_%03d" % k)
        os.makedirs(case_dir, exist_ok=True)
        comm_path = os.path.join(case_dir, "frf.comm")
        with open(comm_path, "w") as fp:
            fp.write(_render_template(
                template_path,
                input_node=in_node, output_node=out_node,
                input_dir=in_dof, output_dir=out_dof,
                f_min=f_min, f_max=f_max, f_step=f_step,
                n_modes=n_modes, damping=damping,
                young=young, poisson=poisson, density=density,
                rails_group=rails_group, locked_dofs=locked_dofs,
            ))
        med_in_local = os.path.join(case_dir, "building.med")
        shutil.copyfile(med_path, med_in_local)
        mess = os.path.join(case_dir, "frf.mess")
        resu = os.path.join(case_dir, "frf.resu")
        med_out = os.path.join(case_dir, "frf.med")
        export_path = _write_export(case_dir, comm_path, med_in_local,
                                     med_out, resu, mess, resu)
        _run_aster(export_path)

        f, H = _parse_table(resu, out_dof)
        if freq_array is None:
            freq_array = f
        elif not np.allclose(freq_array, f):
            raise RuntimeError("frequency axis differs between cases")

        # accelerance from displacement: a = -omega^2 x => H_acc = -omega^2 H_x
        omega = 2.0 * np.pi * f
        H_acc = -omega ** 2 * H
        # millimetre per second^2 per newton (pymodal default unit)
        H_acc_mm = H_acc * 1000.0
        measurements.append(H_acc_mm.reshape(-1, 1, 1))
        coordinates.append((in_coord, out_coord))
        orientations.append((di.upper(), do.upper()))

    if not keep_work and work_dir is None:
        shutil.rmtree(work_root, ignore_errors=True)

    if names is None:
        names = ["frf_%03d" % k for k in range(len(measurements))]

    return frf(
        measurements=measurements if len(measurements) > 1 else measurements[0],
        freq_array=freq_array,
        names=names if len(measurements) > 1 else None,
        name=names[0] if len(measurements) == 1 else None,
        method="SIMO",
        path=path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs=3, type=float, required=True,
                    metavar=("X", "Y", "Z"),
                    help="input excitation point (m)")
    p.add_argument("--output", nargs=3, type=float, default=None,
                    metavar=("X", "Y", "Z"),
                    help="output measurement point (m). Defaults to input.")
    p.add_argument("--input-dir",  default='X', choices=list("XYZxyz"))
    p.add_argument("--output-dir", default='X', choices=list("XYZxyz"))
    p.add_argument("--f-min",  type=float, default=P.F_MIN)
    p.add_argument("--f-max",  type=float, default=P.F_MAX)
    p.add_argument("--f-step", type=float, default=P.F_STEP)
    p.add_argument("--n-modes", type=int, default=P.N_MODES)
    p.add_argument("--damping", type=float, default=P.MODAL_DAMPING)
    p.add_argument("--mesh-dir", default=_HERE)
    p.add_argument("--out", default=None,
                    help="optional .h5 file to persist the resulting frf collection")
    p.add_argument("--keep-work", action="store_true")
    args = p.parse_args()

    H = extract_frf(
        input_xyz=tuple(args.input),
        input_dir=args.input_dir,
        output_xyz=tuple(args.output) if args.output else None,
        output_dir=args.output_dir,
        f_min=args.f_min, f_max=args.f_max, f_step=args.f_step,
        n_modes=args.n_modes, damping=args.damping,
        mesh_dir=args.mesh_dir,
        keep_work=args.keep_work,
        path=args.out,
    )
    print("FRF computed: %d frequencies, item shape %s"
          % (H.freq_array.size, H.measurements[0].shape))
    print("Stored at %s" % H.path)


if __name__ == "__main__":
    _cli()
