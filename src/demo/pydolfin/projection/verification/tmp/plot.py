# Load mayavi
from os import system
from mayavi import *
import string
import time


def plot(file):

    # Plot solution
    v = mayavi()

    v.renwin.renwin.SetOffScreenRendering(1)

    d1 = v.open_vtk_xml(file, config = 0)
    m1 = v.load_module("Axes", config = 0)
    m2 = v.load_module("BandedSurfaceMap", config = 0)
    m3 = v.load_module("Text", config = 0)
    m3.act.SetInput(file[0:(len(file) - 4)])
    m3.act.ScaledTextOff()
    m3.act.GetTextProperty().SetFontSize(30)
    #m3.act.SetWidth(0.4)
    #m3.act.SetHeight(0.1)
    c = m3.act.GetPositionCoordinate()
    c.SetValue(0.0, 0.95)

    # Enable Scalar Legend (a bit messy)
    dvm = v.get_current_dvm()
    mm = dvm.get_current_module_mgr ()
    slh = mm.get_scalar_lut_handler ()

    slh.legend_on.set(1)
    slh.legend_on_off ()

    camera = v.get_render_window().GetActiveCamera()
    
    camera = v.renwin.camera
    camera.Zoom(1.0)
    camera.SetPosition(0.5, 0.4, 3.0)
    camera.SetFocalPoint(0.5, 0.4, 0.5)
    camera.SetViewUp(0.0, 1.0, 0.0)

    v.show_ctrl_panel(0)

    v.Render()


    v.renwin.save_png(file[0:(len(file) - 4)] + ".png")

    # Wait until window is closed
    #v.master.wait_window()

    v.quit()

files = ["projection000000.vtu",
         "interpolation000000.vtu",
         "projection_error000000.vtu",
         "interpolation_error000000.vtu",
         "function000000.vtu",
         "difference000000.vtu"]

for a in files:
    plot(a)

fpng = lambda x: x.__getslice__(0, len(x) - 4) + ".png"

pngfiles = string.join(map(fpng, files))

system("montage -mode Concatenate -tile 2x3 " + pngfiles + " plot.png")
