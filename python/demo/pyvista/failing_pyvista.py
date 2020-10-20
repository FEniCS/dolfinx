import pyvista as pv
import vtk
import numpy as np


def activate_virtual_framebuffer():
    '''
    See: https://github.com/pyvista/pyvista/issues/155

    Activates a virtual (headless) framebuffer for rendering 3D
    scenes via VTK.

    Most critically, this function is useful when this code is being run
    in a Dockerized notebook, or over a server without X forwarding.

    * Requires the following packages:
      * `sudo apt-get install libgl1-mesa-dev xvfb`
    '''

    import subprocess
    import os
    pv.OFFSCREEN = True
    os.environ['DISPLAY'] = ':99.0'

    commands = ['Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &',
                'sleep 3',
                'exec "$@"']

    for command in commands:
        subprocess.call(command, shell=True)


activate_virtual_framebuffer()
print(pv.Report())
cells = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7])
cell_type = np.array([vtk.VTK_LAGRANGE_HEXAHEDRON])
points = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
)


# create the unstructured grid directly from the numpy arrays
grid = pv.UnstructuredGrid(cells, cell_type, points)

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(grid, color="orange", show_edges=True)
plotter.show(screenshot="test.png")
