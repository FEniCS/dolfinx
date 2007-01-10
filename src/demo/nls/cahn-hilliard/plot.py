# Load mayavi
from mayavi import *

# Plot solution
v = mayavi()
d = v.open_vtk_xml("cahn_hilliard000000.vtu")
m = v.load_module("BandedSurfaceMap", 0)

m.contour_on.set(0)

# Set view
v.renwin.z_plus_view()

# Set contour range
dvm = v.get_current_dvm()
mm = dvm.get_current_module_mgr()
mm.scalar_lut_h.range_on_var.set(1)
mm.scalar_lut_h.range_var.set((0, 1))
mm.scalar_lut_h.set_range_var()
mm.scalar_lut_h.sc_bar.SetTitle("Concentration")
 
# Turn on legend
slh = mm.get_scalar_lut_handler ()
slh.legend_on.set(1)
slh.legend_on_off ()

# Wait until window is closed
v.master.wait_window()
