from mpi4py import MPI

import numpy as np

import basix.ufl
import dolfinx
import ufl


def test_compiled_form():
    """
    Compile a form without an associated mesh and assemble a form over a sequence of meshes
    """
    c_el = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    domain = ufl.Mesh(c_el)
    el = basix.ufl.element("Lagrange", "triangle", 2)
    V = ufl.FunctionSpace(domain, el)
    u = ufl.Coefficient(V)
    w = ufl.Coefficient(V)
    c = ufl.Constant(domain)
    e = ufl.Constant(domain)
    J = c * e * u * w * ufl.dx(domain=domain)

    # Compile form using dolfinx.jit.ffcx_jit
    compiled_form = dolfinx.fem.compile_form(MPI.COMM_WORLD, J)

    def create_and_integrate(N, compiled_form):
        mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
        el_2 = basix.ufl.element("Lagrange", "triangle", 2)
        Vh = dolfinx.fem.functionspace(mesh, el_2)
        uh = dolfinx.fem.Function(Vh)
        uh.interpolate(lambda x: x[0])
        wh = dolfinx.fem.Function(Vh)
        wh.interpolate(lambda x: x[1])
        eh = dolfinx.fem.Constant(mesh, 3.0)
        ch = dolfinx.fem.Constant(mesh, 2.0)
        form = dolfinx.fem.create_form(compiled_form, mesh, {u: uh, w: wh}, {c: ch, e: eh})
        assert np.isclose(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM), 1.5)

    # Create various meshes, that all uses this compiled form with a map from ufl
    # to dolfinx functions and constants
    for i in range(1, 4):
        create_and_integrate(i, compiled_form)
