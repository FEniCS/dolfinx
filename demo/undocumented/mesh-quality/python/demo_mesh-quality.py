"This demo illustrates basic inspection of mesh quality."

# Copyright (C) 2013 Jan Blechta
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2013-11-19
# Last changed:

from __future__ import print_function
from dolfin import *

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
interactive()
