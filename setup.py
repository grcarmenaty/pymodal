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
    long_description_content_type="text/markdown",
    url="https://github.com/grcarmenaty/pymodal",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License ",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)