import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymodal",
    version="0.0.1",
    author="Guillermo Reyes Carmenaty",
    author_email="grcarmenaty@gmail.com",
    description="Modal analysis data management and storage tool",
    long_description=long_description,
    url="https://github.com/grcarmenaty/pymodal",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License ",
        "Operating System :: OS Independent",
    ],
    # Should be as lax as possible
    install_requires=[
        "numpy>=1.18.1",
        "scipy>=1.4.1",
        "matplotlib>=3.1.3",
        "pandas>=1.0.1",
        "papergraph>=0.0.1",
    ],
    # Should be as specific as possible
    extras_require={
        "dev": [
            "pytest>=3.7",
            "docutils>=0.16",
            "doc8>=0.8.0",
            "flake8>=3.7.9",
        ],
    },
    python_requires='>=3.6',
)
