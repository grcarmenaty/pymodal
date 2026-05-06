# MCP Server

pymodal ships a [Model Context Protocol](https://modelcontextprotocol.io) server that exposes collection construction, signal processing, FRF computation, SHM-indicator math, and PyTorch-dataset hand-off as tools an LLM agent can call directly. Source: `pymodal/mcp/`.

## When to use the MCP server

Use the MCP server when you want an LLM (Claude Desktop, Claude Code, or any MCP-aware client) to drive an end-to-end pymodal pipeline — *load arrays from disk → assemble HDF5 collection → augment / resample → compute FRFs → compute indicator → split → torch-ready file* — without writing the orchestration code yourself.

When you are writing the pipeline by hand, the MCP server adds nothing — call the Python API directly. The MCP layer is a thin, stateless wrapper that takes file paths in and writes file paths out.

## Install

```bash
pip install -e .[mcp]
```

This adds `mcp>=1.0`. The server is implemented on top of `mcp.server.fastmcp.FastMCP`.

## Run

```bash
python -m pymodal.mcp
# or, after install:
pymodal-mcp
```

Both entry points run the server over stdio, which is the transport local MCP clients (Claude Desktop, Claude Code) expect when they spawn the server as a subprocess.

## Configure your MCP client

Point your client's config at `python -m pymodal.mcp`. For Claude Desktop the relevant snippet of `claude_desktop_config.json` is:

```json
{
  "mcpServers": {
    "pymodal": {
      "command": "python",
      "args": ["-m", "pymodal.mcp"]
    }
  }
}
```

Adjust `command` to the Python interpreter that has pymodal installed (a venv path is fine).

## Design philosophy

- **Stateless.** No in-memory collection objects survive between calls. Every tool takes file paths in and writes a fresh HDF5 file out.
- **Idempotent and composable.** Repeated calls with the same arguments produce the same files; outputs of one tool feed straight into inputs of the next.
- **Disk is the source of truth.** Numpy arrays are never serialised through MCP — heavy data lives on disk, only metadata flows through the protocol.
- **Discovery built-in.** `list_indicators` and `list_collection_classes` return the catalogue of available operations so the agent can plan without looking at the source.

## Typical agent flow

```
create_timeseries_collection
        ↓
add_gaussian_noise          ← optional augmentation
        ↓
timeseries_to_frf           ← also embeds excitation/response references
        ↓
compute_indicator           ← cfdac, sci, rvac, …
        ↓
split_collection            ← stratified train/val/test indices
        ↓
torch_dataset_summary       ← confirm the file is consumable by HDF5Dataset
```

Every step writes a new HDF5 file at the supplied `output_path` and returns a description of the result.

## Tool catalogue

### Discovery

| Tool | Description |
|---|---|
| `describe_collection(path)` | Metadata summary (n_items, item_shape, channel layout, domain axes, labels, references, attached files). |
| `list_collection_classes()` | Every concrete collection class pymodal exposes, grouped by rank. |
| `list_indicators()` | Every SHM indicator method on `frf`, grouped by rank, with a one-line description. |

### Construction

| Tool | Description |
|---|---|
| `create_frf_collection(data_paths, output_path, freq_*, ...)` | Build an `frf` HDF5 from `.npy`/`.npz`/`.mat` arrays on disk. |
| `create_timeseries_collection(data_paths, output_path, sampling_rate \| time_step, ...)` | Build a `timeseries` HDF5 from arrays on disk. Use `method="excitation"` for force inputs. |
| `append_to_collection(input_path, output_path, data_path, name, label, kind)` | Copy a collection and append one extra item. |
| `synthetic_frf(output_path, min_freq, max_freq, resolution, natural_frequencies, damping_ratios, ...)` | Generate a 1-item synthetic FRF for smoke-testing pipelines. |

### Domain edition

| Tool | Description |
|---|---|
| `change_freq_resolution(input_path, output_path, new_resolution)` | Resample every FRF in the collection. |
| `change_freq_span(input_path, output_path, new_min_freq, new_max_freq)` | Crop or extend every FRF along the frequency axis. |
| `change_sampling_rate(input_path, output_path, new_sampling_rate)` | Resample every signal in a `timeseries` collection. |
| `change_time_span(input_path, output_path, new_min_time, new_max_time)` | Crop or extend every signal along the time axis. |

### Augmentation

| Tool | Description |
|---|---|
| `add_gaussian_noise(input_path, output_path, min_amplitude, max_amplitude, sample_fraction)` | Append Gaussian-noise-augmented copies (with `_augmented` suffix) to a `timeseries` collection. Originals are preserved. |

### Time → frequency

| Tool | Description |
|---|---|
| `timeseries_to_frf(response_path, excitation_path, output_path, frf_type="H1", resp_delay=0)` | Compute FRFs from a response/excitation pair. The result embeds both inputs as references (`"input"`, `"output"`). |

### Indicators

| Tool | Description |
|---|---|
| `compute_indicator(indicator, reference_path, damaged_path, output_path, std=6.0)` | Compute any of the 16 SHM indicators (`sci`, `cfdac`, `rvac`, …) for a (reference, damaged) FRF pair. The result embeds both inputs as references. `std` is only used by `frfsm`. |

### Splitting and PyTorch

| Tool | Description |
|---|---|
| `split_collection(path, train_frac, val_frac, test_frac, seed, kind)` | Stratified train/val/test index split. Returns indices and stores them on the collection. `kind` is `"frf"` or `"timeseries"`. |
| `torch_dataset_summary(path)` | Open the file as `HDF5Dataset` and report sample count, first-sample shape and dtype, and label presence. Confirms the file is consumable by a `DataLoader`. |

## What gets returned

Every tool that produces a new collection also calls `describe_collection` on the result and returns the description. The agent doesn't need a follow-up `describe_collection` call to know what it just built.

A typical description looks like:

```json
{
  "path": "/abs/path/cfdac.h5",
  "class": "cfdac_collection",
  "n_items": 12,
  "item_shape": [1024, 1024, 1, 1],
  "method": "SIMO",
  "n_outputs": 1,
  "n_inputs": 1,
  "labels": [1.0, 1.0, 1.0, ...],
  "references": ["reference", "damaged"],
  "domain_axes": [
    {"name": "axis_0", "length": 1024, "min": 0.0, "max": 1023.0, "units": "dof_index"},
    {"name": "axis_1", "length": 1024, "min": 0.0, "max": 1023.0, "units": "dof_index"}
  ]
}
```

## Pitfalls

- **The server is stateless — every tool reads from disk.** That means it is fine to interleave calls from several conversations, but it also means that an `output_path` overwrites whatever was at that path before. Pick distinct paths for distinct artifacts.
- **The MCP layer accepts only the formats `pymodal.load_array` understands** for input arrays: `.npy`, `.npz`, `.mat`. Convert other formats first.
- **`compute_indicator` requires both `reference_path` and `damaged_path` to point at FRF collections** — not time-series. Run `timeseries_to_frf` first.
- **`split_collection` requires labels** for stratification to be meaningful. Without labels you'll get all items in one bucket.
- **The server uses absolute paths internally** (resolved via `Path.expanduser().resolve()`). Relative paths in client requests are resolved against the server's working directory.
