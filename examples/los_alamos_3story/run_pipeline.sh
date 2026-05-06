#!/usr/bin/env bash
# Convenience driver for the Los Alamos 3-storey benchmark example.
#
# Usage:
#   ./run_pipeline.sh build                       # geometry + mesh
#   ./run_pipeline.sh frf  IX IY IZ  OX OY OZ ID OD  [extra-args...]
#                                                 # extract one FRF
#
# IX/IY/IZ      input excitation point (m)
# OX/OY/OZ      output measurement point (m)
# ID/OD         input/output directions ('X', 'Y', or 'Z')
# extra-args    forwarded verbatim to frf_extract.py (e.g. --f-max 400)
#
# Override environment defaults:
#   SALOME_BIN     command used to run Salome (default: 'salome')
#   ASTER_RUNNER   command used to run Code_Aster (default: 'as_run')
#   PYTHON         interpreter used for frf_extract.py (default: 'python')
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SALOME_BIN="${SALOME_BIN:-salome}"
PYTHON="${PYTHON:-python}"
export ASTER_RUNNER="${ASTER_RUNNER:-as_run}"

cmd="${1:-}"
shift || true

case "$cmd" in
    build)
        echo "[build] running Salome to build geometry and mesh..."
        "$SALOME_BIN" -t "$HERE/salome_build.py"
        ;;
    frf)
        if [ "$#" -lt 8 ]; then
            echo "Usage: $0 frf IX IY IZ OX OY OZ ID OD [extra-args...]" >&2
            exit 2
        fi
        IX="$1"; IY="$2"; IZ="$3"
        OX="$4"; OY="$5"; OZ="$6"
        ID="$7"; OD="$8"
        shift 8
        "$PYTHON" "$HERE/frf_extract.py" \
            --input  "$IX" "$IY" "$IZ" \
            --output "$OX" "$OY" "$OZ" \
            --input-dir  "$ID" \
            --output-dir "$OD" \
            "$@"
        ;;
    *)
        echo "Usage: $0 {build|frf} [...]" >&2
        echo "  $0 build" >&2
        echo "  $0 frf IX IY IZ OX OY OZ ID OD [extra args]" >&2
        exit 2
        ;;
esac
