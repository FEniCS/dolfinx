# DOLFINx Python interface installation

Below is guidance for building the DOLFINx Python interface.

1. Build and install the DOLFINx C++ library.

2. Ensure the Python interface build requirements are installed:

       pip install scikit-build-core
       python -m scikit_build_core.build requires | python -c "import sys, json; print(' '.join(json.load(sys.stdin)))" | xargs pip install

3. Build DOLFINx Python interface:

       pip install --check-build-dependencies --no-build-isolation .

To build in Developer and editable mode for development:

     pip -v install --check-build-dependencies -Cbuild-dir="build" -Ccmake.build-type="Developer" -Cinstall.strip=false --no-build-isolation -e .

Note that Developer mode is significantly stricter than CMake's default Debug mode.

# Type checking with mypy

1. Install DOLFINx Python with the `[mypy]` optional dependencies set, e.g.:

       pip install .[mypy]

2. Check with mypy, e.g.:

       mypy --config-file pyproject.toml -p dolfinx

   The `--config-file pyproject.toml` is mandatory to run mypy with the correct options.
   The `-p` flag checks the built/installed package `dolfinx`, containing the C++
   bindings and Python interface.
