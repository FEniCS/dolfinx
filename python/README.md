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

# Type checking with Pyrefly

1. Install DOLFINx Python with the `typing` extra, plus Pyrefly itself
   (or any other type checker), e.g.:

       pip install pyrefly '.[typing]'

2. Check with Pyrefly:

       pyrefly check

   Run this command from the `python` directory. The `pyproject.toml` configuration
   checks `dolfinx`, `demo`, and `test`, using the built/installed package to resolve
   the C++ bindings.
