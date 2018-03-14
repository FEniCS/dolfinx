"This demo illustrates basic inspection of mesh quality."

# Copyright (C) 2013 Jan Blechta
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from dolfin import *
import matplotlib.pyplot as plt


# Read mesh from file
mesh = Mesh("../dolfin_fine.xml.gz")

# Print minimal and maximal radius ratio
qmin, qmax = MeshQuality.radius_ratio_min_max(mesh)
print('Minimal radius ratio:', qmin)
print('Maximal radius ratio:', qmax)

# Show histogram using matplotlib
hist = MeshQuality.radius_ratio_matplotlib_histogram(mesh)
hist = hist.replace('    import matplotlib.pylab', '    import matplotlib\n    matplotlib.use(\'Agg\')\n    import matplotlib.pylab\n')
hist = hist.replace('matplotlib.pylab.show()', 'matplotlib.pylab.savefig("mesh-quality.pdf")')
print(hist)
exec(hist)

# Show mesh
plot(mesh)
plt.show()
