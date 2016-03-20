#!/usr/bin/env py.test
import ufl
from dolfin import *
from dolfin_utils.test import skip_in_parallel


@skip_in_parallel
def test_interior_facet_integral_sides():
    n = 1
    mesh = UnitSquareMesh(n, n)
    markers = CellFunctionSizet(mesh)
    subdomain = AutoSubDomain(lambda x, on_boundary: x[0] > x[1]-DOLFIN_EPS)

    V = FunctionSpace(mesh, "DG", 0)
    f = interpolate(Expression("x[0]", degree=1), V)

    # Define forms picking value of f from + and - sides
    scale = 1.0/ufl.FacetArea(mesh)
    Mp = f('+')*scale*dS(domain=mesh)
    Mm = f('-')*scale*dS(domain=mesh)

    # Hack to attach cell markers to dS integral... Need to find a UFL
    # solution to this.
    Mh = Constant(0)*dx(99, domain=mesh, subdomain_data=markers)
    Mp = Mp + Mh
    Mm = Mm + Mh

    # Case A: subdomain is 1, rest is 0
    markers.set_all(0)
    subdomain.mark(markers, 1)
    assert abs(assemble(Mp) - 2.0/3.0) < 1e-8
    assert abs(assemble(Mm) - 1.0/3.0) < 1e-8

    # Case B: subdomain is 0, rest is 1
    markers.set_all(1)
    subdomain.mark(markers, 0)
    assert abs(assemble(Mp) - 1.0/3.0) < 1e-8
    assert abs(assemble(Mm) - 2.0/3.0) < 1e-8
