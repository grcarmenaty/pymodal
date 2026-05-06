=================
PyModal Library
=================
--------------------------------------------------------------------------------
Labelled HDF5-backed vibrational-signal collections, ready for PyTorch SHM work
--------------------------------------------------------------------------------

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/grcarmenaty/pymodal/blob/master/demo.ipynb
   :alt: Open the end-to-end demo in Colab

Introduction
============

``pymodal`` is a research library developed by the Applied Mechanics Lab,
IQS School of Engineering, for Structural Health Monitoring (SHM) work.
Its primary purpose is to make it easy to build large, labelled
collections of vibrational signals stored on disk, and to feed those
collections directly into PyTorch ML training pipelines — without
loading the dataset into RAM.

The intended workflow is:

1. Acquire or simulate vibrational measurements (time-series or FRFs).
2. Assemble them into a ``timeseries`` or ``frf`` collection (a single
   ``ndarray`` builds a 1-item collection, a list of ``ndarray`` s
   builds a multi-item collection). Every measurement array is written
   to an HDF5 file on disk as it is added.
3. Pre-process the collection in-place (resample, change span, augment
   with noise, convert to FRF) — all operations stream through the
   HDF5 file without materialising the full dataset in memory.
4. Optionally compute SHM indicators (CFDAC, RVAC, SCI, …) — each
   returns a typed indicator collection (``cfdac_collection``,
   ``sci_collection``, …) with the source ``reference`` and ``damaged``
   collections embedded as references for full provenance.
5. Call ``.torch_dataset()`` to obtain an ``HDF5Dataset``
   (``torch.utils.data.Dataset``) that lazy-loads individual samples on
   demand during training.

Signal processing, metrics, simulation, and plotting are secondary
capabilities that support building and validating those collections.

Features
--------

- Dimensional collection hierarchy: ``_collection_0d`` / ``_collection_1d``
  / ``_collection_2d`` / ``_collection_3d``, with ``frf`` and
  ``timeseries`` as ``_collection_1d`` subclasses.
- HDF5 persistence layer with on-disk storage and lazy access.
- First-class numeric labels per item, surfaced as ``y`` in PyTorch
  ``(x, y)`` batches.
- First-class reference collections, embedded by copy, so consumer
  files stay self-contained even when the source moves or is deleted.
- FRF estimation from time-domain data via ``pyFRF`` (H1, H2, Hv,
  vector, ODS).
- SHM damage-detection metrics: CFDAC, FDAC, RVAC, GAC, SCI, DRQ,
  AIGAC, FRFRMS, FRFSF, FRFSM, M2L, …, each available both as a pure
  function in ``pymodal.utils`` and as a typed indicator collection
  class.
- Synthetic FRF and time-series generation via modal superposition.
- Data augmentation: Gaussian noise injection (``AddGaussianNoise``)
  via ``audiomentations``.
- Optional `MCP <https://modelcontextprotocol.io>`_ server that exposes
  the full pipeline as tools an LLM agent can call directly.

Installation
============

.. code-block:: bash

    pip install git+https://github.com/grcarmenaty/pymodal.git@master

For local development, clone the repo and run

.. code-block:: bash

    pip install -e .[dev]

The MCP-server extras are installed with

.. code-block:: bash

    pip install -e .[mcp]

``pymodal`` requires Python ≥ 3.9 and pulls in ``numpy``, ``scipy``,
``matplotlib``, ``pandas``, ``Pint``, ``pyFRF``, ``h5py``,
``audiomentations``, and ``torch`` automatically.

Notebooks
=========

Every notebook below carries an *Open in Colab* badge and a Colab-aware
setup cell at the top, so it runs end-to-end on a fresh Colab runtime
with no manual configuration. They also run unchanged from a local
clone.

Library walkthroughs
--------------------

- ``demo.ipynb`` — End-to-end demo

  .. image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://colab.research.google.com/github/grcarmenaty/pymodal/blob/master/demo.ipynb
     :alt: Open in Colab

  Full pipeline from synthetic SDOF accelerance FRFs to a trained
  PyTorch classifier: ``frf`` and ``timeseries`` construction, batch
  processing, HDF5 persistence, ``HDF5Dataset`` + ``DataLoader``,
  1-D CNN training with gradient accumulation and mixed precision, and
  a 2-D CNN over a ``cfdac_collection`` indicator.

- ``architecture.ipynb`` — Dimensional collections, references, typed indicators

  .. image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://colab.research.google.com/github/grcarmenaty/pymodal/blob/master/architecture.ipynb
     :alt: Open in Colab

  Didactic tour of the post-refactor architecture (no model training).
  Covers the four dimensional parents, single- vs multi-array
  constructors, the on-disk HDF5 layout, domain-edition methods,
  embed-by-copy reference linking, source-survival, ``timeseries.to_FRF``
  reference auto-registration, typed indicators (0-D / 1-D / 2-D),
  provenance round-trip, and the PyTorch hand-off for variable-rank
  items.

End-to-end SHM examples
-----------------------

- ``examples/los_alamos_3story/los_alamos_demo.ipynb`` — LANL 3-storey building

  .. image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://colab.research.google.com/github/grcarmenaty/pymodal/blob/master/examples/los_alamos_3story/los_alamos_demo.ipynb
     :alt: Open in Colab

  Full pymodal pipeline on the LANL 3-storey benchmark with a
  rail-constrained, screw-bolted geometry. 49 damage scenarios
  (1 pristine + 12 columns × 4 thinning levels), a 4 900-sample
  labelled ``frf`` collection, and three classification heads
  (detection, localisation, severity) trained from the same HDF5 file.

- ``examples/cantilever_crack/cantilever_demo.ipynb`` — Cantilever beam crack diagnosis

  .. image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://colab.research.google.com/github/grcarmenaty/pymodal/blob/master/examples/cantilever_crack/cantilever_demo.ipynb
     :alt: Open in Colab

  Same three-head SHM workflow, applied to a 0.5 m steel cantilever
  with a simulated crack. 25 damage scenarios over 6 positions × 4
  depths, a 2 500-sample labelled ``frf`` collection, and three heads:
  detection (classification), crack localisation (regression on
  distance from the fixed end), severity (regression on crack depth).

- ``examples/transfer_learning/transfer_learning_demo.ipynb`` — Transfer learning across structures

  .. image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://colab.research.google.com/github/grcarmenaty/pymodal/blob/master/examples/transfer_learning/transfer_learning_demo.ipynb
     :alt: Open in Colab

  Reuses a damage-detection model trained on the LANL benchmark as the
  initialisation for the cantilever detection head, sharing nothing
  beyond "linear vibrations measured as FRFs". Demonstrates the
  ``coll.save_model`` / ``coll.load_model`` API that lets each ``frf``
  collection carry its trained heads alongside its data, and compares
  fine-tuned transfer to a from-scratch baseline.

Quick start
===========

Build a labelled FRF collection and feed it to PyTorch:

.. code:: python

    import numpy as np
    import pymodal

    freq = np.arange(0.0, 200.5, 0.5)            # 0–200 Hz, 0.5 Hz step
    arrays = [np.random.randn(len(freq), 4, 1) for _ in range(50)]   # 4 outputs, 1 input
    labels = np.random.randint(0, 3, size=50)                        # 3 damage classes

    # Single HDF5-backed collection on disk.
    coll = pymodal.frf(
        measurements=arrays,
        freq_array=freq,
        path="train.h5",
        labels=labels.tolist(),
        method="SIMO",
    )

    # In-place processing.
    coll.change_freq_resolution(1.0)              # downsample to 1 Hz
    coll.change_freq_span(20.0, 150.0)             # crop to a band of interest

    # PyTorch hand-off — items are loaded lazily from train.h5.
    dataset = coll.torch_dataset()
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

Compute a typed indicator collection and recover provenance:

.. code:: python

    cfdac = damaged.cfdac(reference)
    cfdac.references["reference"]                   # the source healthy collection
    cfdac.references["damaged"]                     # the source damaged collection
    cfdac.get_reference_data("reference", i=0)      # array for item 0, read from the consumer's HDF5

Damage indicators
=================

The indicator catalogue lives both as pure functions in ``pymodal.utils``
and as typed collection classes. Pick the dimensionality that matches
the indicator output:

- 0-D scalars per item — ``sci``, ``unsigned_sci``, ``drq``, ``aigac``,
  ``frfrms``, ``frfsf``, ``frfsm``, ``ods_diff``, ``r2_imag``.
- 1-D vectors per item — ``rvac``, ``rvac_2d``, ``gac``, ``m2l``.
- 2-D matrices per item — ``cfdac``, ``cfdac_a``, ``fdac``.

Every indicator computed from a pair of FRF collections registers the
``reference`` and ``damaged`` collections as embedded references on the
result, so the consumer file is fully self-contained.

MCP server
==========

``pymodal`` ships an `MCP <https://modelcontextprotocol.io>`_ server
that exposes collection construction, signal processing, FRF
computation, SHM-indicator math, and PyTorch-dataset hand-off as tools
an LLM agent can call directly.

Install the optional dependency:

.. code-block:: bash

    pip install -e .[mcp]

Run the server (stdio transport):

.. code-block:: bash

    python -m pymodal.mcp
    # or, after install:
    pymodal-mcp

Configure your MCP client (Claude Desktop, Claude Code, …) to launch
``python -m pymodal.mcp`` as a server. Every tool takes file paths in
and writes file paths out, so a typical agent flow is

``create_timeseries_collection`` → ``add_gaussian_noise`` →
``timeseries_to_frf`` → ``compute_indicator`` → ``split_collection`` →
``torch_dataset_summary``.

Use ``list_indicators`` and ``list_collection_classes`` for in-tool
discovery.

Tests
=====

.. code-block:: bash

    pytest tests/

Five test files, 84 tests total; CI exercises Python 3.9 and 3.10 on
Ubuntu, macOS and Windows.
