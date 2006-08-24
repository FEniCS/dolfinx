# Plot using Mayavi

import mayavi
#v = mayavi.mayavi(geometry="743x504") # create a MayaVi window.
v = mayavi.mayavi() # create a MayaVi window.
d = v.open_vtk_xml("solution000001.vtu", config=0) # open the data file.
# The config option turns on/off showing a GUI control for the
# data/filter/module.

# Load the filters.
f = v.load_filter('WarpScalar', config=0)
f.fil.SetScaleFactor(0.2)
n = v.load_filter('PolyDataNormals', 0)
n.fil.SetFeatureAngle(45) # configure the normals.

# Load the necessary modules.
m = v.load_module('SurfaceMap', 0)
a = v.load_module('Axes', 0)
a.axes.SetCornerOffset(0.0) # configure the axes module.
a.axes.SetFontFactor(1.0) # configure the axes module.
o = v.load_module('Outline', 0)

camera = v.renwin.camera
camera.Zoom(1.0)

# Re-render the scene.
v.Render()


# Animation
# Is there a cleaner way to perform sweeping through a time series?

def myanim(v, d, f):
    v.renwin.save_png('solution%4.4d.png' % d.get_timestep())
    N = len(d.get_file_list())
    i = d.get_timestep() + 1
    d.set_timestep(i % N)
    dvm = v.get_current_dvm()
    mm = dvm.get_current_module_mgr()
    mm.update_modules()
    v.Render()

v.start_animation(400, myanim, v, d, f)

# Comment out if you run "python -i"
v.master.wait_window()
