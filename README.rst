=================
PyModal Library
=================
------------------------------------------------------
Simulate, load, store and represent your modal data
------------------------------------------------------

Introduction
============

This library is a work in progress developed by the Applied Mechanics Lab, 
IQS School of Engineering, as part of the research in Structural Health Monitoring (SHM). 
This library enables the user to store and process Frequency Response Functions (FRF). 
The library includes a wide range of frequency-based damage indicators found in the literature. 
Currently, Transmissibility Functions (TF) are being implemented in the library.
The library also includes tools for building certain geometries in ANSYS and get their FRFs. 
If you plan on using the ANSYS module, be sure to have a working ANSYS installation.

Installation
============

In order to install this module, just run

.. code-block:: bash
    
    pip install pymodal

in your terminal. This will also potentially install all the requirements, which
you can find in `requirements.txt <https://github.com/grcarmenaty/pymodal/blob/master/requirements.txt>`_, although they will be included here as
well for clarity's sake:

- numpy
- scipy
- matplotlib
- pandas
- pyansys

Dev Installation
----------------

If you wish to try and add some features yourself or modify some of the existing
ones, clone the repository and, in the same folder where the repo is cloned,
run the following command:

.. code-block:: bash
    
    pip install -e .[dev]

This will also potentially install all the development requirements, which
you can find in `requirements-dev.txt <https://github.com/grcarmenaty/pymodal/blob/master/requirements-dev.txt>`_, although they are included here as
well for clarity's sake:

- pytest
- docutils
- doc8
- flake8

Basic usage
============

Make an instance of ``FRF`` class:
----------------

At least one of the following must be specified: Resolution, Bandwidth or Maximum frequency.  
If not specified, minimum frequency is assumed to be 0 Hz.

.. code:: python

   a = pymodal.FRF(
       frf,
       resolution,
       bandwidth,
       max_freq,
       min_freq,
       name,
       part,
       modal_frequencies
       )
       
Add new FRFs to existing instance:
----------------


Save instance as zip file:
----------------


Make an instance of ``FRF`` class:
----------------

Post-processing FRFs
============

Frequency-based damage indicators
============
Currently, the pymodal library holds the following damage indicators:

- Frequency Response Function RMS [FRFRMS](https://www.sciencedirect.com/science/article/abs/pii/S1270963802011938)
- Global Amplitude Criterion [GAC]
- Average Integration Global Amplitude Criterion (AIGAC)
- Frequency Domain Assurance Criterion (FDAC)
- Response Vector Assurance Criterion (RVAC)
- Detection and Relative Quantification (DRQ)
- Detection and Relative Quantification curvature-based (DRQ'')
- Frequency Response Function Scale Factor (FRFSF)
- Coefficient of Determination (R^2)
- ∆ODS (Operational Deflection Shapes)
- Frequency Response Function Similarity Metric (FRFSM)
- Complex Frequency Domain Assurance Crietrion [CFDAC](https://www.sciencedirect.com/science/article/abs/pii/S0888327018306551?via%3Dihub)
- Spectral Correlation Index [SCI](https://www.sciencedirect.com/science/article/abs/pii/S0888327018306551?via%3Dihub)

Transmissibility Functions (TF)
============




ANSYS
============
