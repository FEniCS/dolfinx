# DOLFINx Python interface installation

Below is guidance for building the DOLFINx Python interface.

1. Build and install the DOLFINx C++ library.

2. Ensure the Python interface build requirements are installed:

       pip install --group pyproject.toml:build

3. Build DOLFINx Python interface:

       pip install --check-build-dependencies --no-build-isolation .

To build in Developer and editable mode for development:

     pip -v install --check-build-dependencies -Cbuild-dir="build" -Ccmake.build-type="Developer" -Cinstall.strip=false --no-build-isolation -e .

Note that Developer mode is significantly stricter than CMake's default Debug mode.

To build with sanitizers for finding bugs rather than performance, see the
[sanitizer info](../cpp/cmake/sanitizers/README.md).

# Type checking with mypy

1. Install DOLFINx Python with the `typing` extra, plus `mypy` itself
   (or any other type checker), e.g.:

       pip install mypy '.[typing]'

2. Check with mypy, e.g.:

       mypy --config-file pyproject.toml -p dolfinx

   The `--config-file pyproject.toml` is mandatory to run mypy with the correct options.
   The `-p` flag checks the built/installed package `dolfinx`, containing the C++
   bindings and Python interface.
