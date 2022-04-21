# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Lagrange variants
#
# This demo is implemented in a single Python file,
# {download}`demo_lagrange_variants.py`. It illustrates how to:
#
# - Define finite elements directly using Basix
# - Create variants of Lagrange finite elements
#
# ## Lagrange with equispaced points

import numpy as np

import ufl
import basix
import basix.ufl_wrapper
import matplotlib.pylab as plt
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, grad, inner

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

# Degree 10 with equispaced points is bad... Runge's phenomenon.

element = basix.create_element(
    basix.ElementFamily.P, basix.CellType.interval, 10, basix.LagrangeVariant.equispaced)

pts = basix.create_lattice(basix.CellType.interval, 200, basix.LatticeType.equispaced, True)

values = element.tabulate(0, pts)[0, :, :, 0]

for i in range(values.shape[1]):
    plt.plot(pts, values[:, i])
plt.ylim([-1, 6])

plt.savefig("demo_lagrange_variants_equispaced_10.png")
plt.clf()

# ![](demo_lagrange_variants_equispaced_10.png)

# Degree 10 with GLL Points is better...

element = basix.create_element(
    basix.ElementFamily.P, basix.CellType.interval, 10, basix.LagrangeVariant.gll_warped)

pts = basix.create_lattice(basix.CellType.interval, 200, basix.LatticeType.equispaced, True)

values = element.tabulate(0, pts)[0, :, :, 0]

for i in range(values.shape[1]):
    plt.plot(pts, values[:, i])
plt.ylim([-1, 6])

plt.savefig("demo_lagrange_variants_gll_10.png")
plt.clf()

# ![](demo_lagrange_variants_gll_10.png)

# To use elements directly from Basix...


element = basix.create_element(
    basix.ElementFamily.P, basix.CellType.interval, 10, basix.LagrangeVariant.gll_warped)
ufl_element = basix.ufl_wrapper.BasixElement(element)

# Solve a problem

# Compare solution with equispaced variant and GLL variant
