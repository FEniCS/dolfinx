# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Poisson equation
#
# ## Implementation
#
# Testing 123

# +
import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from ufl import ds, dx, exp, grad, inner, sin

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
