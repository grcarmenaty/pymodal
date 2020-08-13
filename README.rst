=================
PyModal Library
=================
------------------------------------------------------
Simulate, load, store and represent your modal data
------------------------------------------------------

Introduction
============

This library is a work in progress dedicated to storing FRFs with similar
sampling frequency and time window. For now it's only built for storing
processed FRFs into numpy arrays with a few extra information alongside them.
It also includes tools for building certain geometries in ANSYS and get their
FRFs. This is still a very early alpha, but the main objective of this project
is to comfortably have FRFs for training deep learning models with ease. If
you plan on using the ANSYS module, be sure to have a working ANSYS
installation.

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
you can find in `requirements-dev.txt <https://github.com/grcarmenaty/pymodal/blob/master/requirements-dev.txt>`_, although they will be included here as
well for clarity's sake:

- pytest
- docutils
- doc8
- flake8