"""MCP server for pymodal.

Exposes pymodal's collection-building, signal-processing, FRF, and SHM
indicator capabilities as Model Context Protocol tools so that an LLM agent
can drive an end-to-end "raw arrays on disk → labelled HDF5 collection →
indicator collection → torch dataset" pipeline.

The server is built on :mod:`mcp.server.fastmcp` and runs over stdio, which
is the transport Claude Desktop / Claude Code expect for local servers:

.. code-block:: bash

    python -m pymodal.mcp

See :mod:`pymodal.mcp.server` for the full tool list.
"""

from pymodal.mcp.server import mcp, main

__all__ = ["mcp", "main"]
