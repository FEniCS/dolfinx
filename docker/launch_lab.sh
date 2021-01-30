#!/bin/bash
set -x
export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_IPYVTK=true
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
set +x
jupyter lab --ip 0.0.0.0 --no-browser --allow-root
exec "$@"