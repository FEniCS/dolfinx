"This demo illustrates basic inspection of mesh quality."

# Copyright (C) 2013-2018 Jan Blechta and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy as np
import matplotlib.pyplot as plt
import dolfin.cpp.mesh
from dolfin import *


# Create mesh
n = 12
mesh = BoxMesh.create(MPI.comm_world, [Point(0, 0, 0), Point(1, 1, 1)], [
                      n, n, n], CellType.Type.tetrahedron)

# Print minimum and maximum radius ratio
qmin, qmax = MeshQuality.radius_ratio_min_max(mesh)
print('Minimal radius ratio:', qmin)
print('Maximal radius ratio:', qmax)

# Plot histogram for radius ratio
bins, num = dolfin.cpp.mesh.MeshQuality.radius_ratio_histogram_data(mesh, 20)
plt.subplot(221)
plt.bar(bins, num, width=0.9*(bins[1]-bins[0]))
plt.title("Radius ratios (original mesh)")
plt.xlabel("ratio")
plt.ylabel("number of cells")
plt.xlim(0.0, 1.0)
plt.xticks((0.0, 0.5, 1.0))

# Plot histogram for diahedral angle
bins, num = dolfin.cpp.mesh.MeshQuality.dihedral_angle_histogram_data(mesh, 20)
plt.subplot(222)
plt.bar(bins, num, width=0.9*(bins[1]-bins[0]))
plt.title("Dihedral angle (original mesh)")
plt.xlabel("angle")
plt.ylabel("number of angles")
plt.xlim(0.0, np.pi)
plt.xticks((0.0, np.pi/2.0, np.pi), (0, '$\pi/2$', '$\pi$'))

# Move points
x = mesh.geometry.points[:]
xr = np.random.randn(x.shape[0], x.shape[1])
x += 0.2*(1.0/n)*xr

# Plot histogram for radius ratio (perturbed mesh)
bins, num = dolfin.cpp.mesh.MeshQuality.radius_ratio_histogram_data(mesh, 20)
plt.subplot(223)
plt.bar(bins, num, width=0.9*(bins[1]-bins[0]))
plt.title("Radius ratios (perturbed mesh)")
plt.xlabel("ratio")
plt.ylabel("number of cells")
plt.xlim(0.0, 1.0)
plt.xticks((0.0, 0.5, 1.0))

# Plot histogram for diahedral angle (perturbed mesh)
bins, num = dolfin.cpp.mesh.MeshQuality.dihedral_angle_histogram_data(mesh, 20)
plt.subplot(224)
plt.bar(bins, num, width=0.9*(bins[1]-bins[0]))
plt.title("Dihedral angle (perturbed mesh)")
plt.xlabel("angle")
plt.ylabel("number of angles")
plt.xlim(0.0, np.pi)
plt.xticks((0.0, np.pi/2.0, np.pi), (0, '$\pi/2$', '$\pi$'))

# Show plot
plt.show()
