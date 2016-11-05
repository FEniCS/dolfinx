
.. _demo_nonmataching_interpolation:

Interpolation from a non-matching mesh
======================================

This example demonstrates how to interpolate functions between
finite element spaces on non-matching meshes.

First, the :py:mod:`dolfin` module is imported: ::

  from dolfin import *

Next, we create two different meshes. In this case we create unit
square meshes with different size cells ::

  mesh0 = UnitSquareMesh(16, 16)
  mesh1 = UnitSquareMesh(64, 64)

On each mesh we create a finite element space. On the coarser mesh we use linear
Lagrange elements, and on the finer mesh cubic Lagrange elements ::

  P1 = FunctionSpace(mesh0, "Lagrange", 1)
  P3 = FunctionSpace(mesh1, "Lagrange", 3)

We interpolate the function :math:`\sin(10x) \sin(10*y))` ::

  v = Expression("sin(10.0*x[0])*sin(10.0*x[1])", degree=5)

into the `P3` finite element space ::

  # Create function on P3 and interpolate v
  v3 = Function(P3)
  v3.interpolate(v)

We now interpolate the function `v3` intto the `P1` space ::

  # Create function on P1 and interpolate v3
  v1 = Function(P1)
  v1.interpolate(v3)

The interpolated functions, `v3` and `v1` can ve visualised using the
`plot` function ::

  plot(v3, title='v3')
  interactive()

  plot(v1, title='v1')
  interactive()
