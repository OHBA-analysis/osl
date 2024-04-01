# OHBA Software Library (OSL)

Tools for MEG/EEG analysis.

Documentation: https://osl.readthedocs.io/en/latest/.

## Installation

See the [official documentation](https://osl.readthedocs.io/en/latest/install.html) for recommended installation instructions.

## For Developers

Run tests:
```
cd osl
pytest tests
```
or to run a specific test:
```
cd osl/tests
pytest test_file_handling.py
```

Build documentation:
```
python setup.py build_sphinx
```
Compiled docs can be found in `doc/build/html/index.html`.
