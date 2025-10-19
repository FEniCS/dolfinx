# DOLFINx Python interface installation

Below is guidance for building the DOLFINx Python interface.

1. Build and install the DOLFINx C++ library.

2. Ensure the Python interface build requirements are installed:

          python -m scikit_build_core.build requires | python -c "import sys, json; print(' '.join(json.load(sys.stdin)))" | xargs pip install

3. Build DOLFINx Python interface:

          pip install --check-build-dependencies --no-build-isolation .

To build in debug and editable mode for development:

     pip -v install --check-build-dependencies --config-settings=build-dir="build" --config-settings=cmake.build-type="Debug"  --config-settings=install.strip=false --no-build-isolation -e .
