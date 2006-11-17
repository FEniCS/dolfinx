# Load mayavi
from mayavi import *

# Plot solution
v = mayavi()
d = v.open_vtk_xml("disp000000.vtu")

f = v.load_filter('WarpVector', config=0) 

# Set scaling
f.max_scale_var.set(100)
f.min_scale_var.set(0)
f.fil.SetScaleFactor(100)

m = v.load_module("BandedSurfaceMap")
m.actor.GetProperty().SetColor((0.8, 0.8, 1.0))

# Camera
camera = v.renwin.camera
camera.Zoom(1.5)
camera.SetPosition(0.5, 0.5, 3.0)
camera.SetRoll(0.0)

# Turn on sweeping
d.sweep_delay.set(0.1)
d.sweep_var.set(1)
d.do_sweep()


# Wait until window is closed
v.master.wait_window()
