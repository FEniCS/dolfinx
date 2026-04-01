import typing
from dataclasses import dataclass

from mpi4py import MPI

import numpy as np
import numpy.typing as npt

import basix.ufl
import dolfinx
import ufl


@dataclass
class GeneralForm:
    ufl_form: ufl.Form
    ufcx_form: typing.Any
    module: typing.Any
    code: str


def compile_form(
    comm: MPI.Intracomm,
    form: ufl.Form,
) -> GeneralForm:
    """Compile UFL form withtout associated DOLFINx data."""
    compiled_form = dolfinx.jit.ffcx_jit(comm, form)
    return GeneralForm(form, *compiled_form)


def get_integration_domains(integral_type, subdomain):
    """Get integration domains from subdomain data."""
    if subdomain is None:
        return []
    else:
        try:
            if integral_type in (
                dolfinx.fem.IntegralType.exterior_facet,
                dolfinx.fem.IntegralType.interior_facet,
            ):
                tdim = subdomain.topology.dim
                subdomain._cpp_object.topology.create_connectivity(tdim - 1, tdim)
                subdomain._cpp_object.topology.create_connectivity(tdim, tdim - 1)
            domains = dolfinx.cpp.fem.compute_integration_domains(
                integral_type, subdomain._cpp_object
            )
            return [(s[0], np.array(s[1])) for s in domains]
        except AttributeError:
            return [(s[0], np.array(s[1])) for s in subdomain]


def form_cpp_creator(
    dtype: npt.DTypeLike,
) -> (
    dolfinx.cpp.fem.Form_float32
    | dolfinx.cpp.fem.Form_float64
    | dolfinx.cpp.fem.Form_complex64
    | dolfinx.cpp.fem.Form_complex128
):
    """Return the wrapped C++ class of a variational form of a specific scalar type.

    Args:
        dtype: Scalar type of the required form class.

    Returns:
        Wrapped C++ form class of the requested type.

    Note:
        This function is for advanced usage, typically when writing
        custom kernels using Numba or C.
    """
    if np.issubdtype(dtype, np.float32):
        return dolfinx.cpp.fem.create_form_float32
    elif np.issubdtype(dtype, np.float64):
        return dolfinx.cpp.fem.create_form_float64
    elif np.issubdtype(dtype, np.complex64):
        return dolfinx.cpp.fem.create_form_complex64
    elif np.issubdtype(dtype, np.complex128):
        return dolfinx.cpp.fem.create_form_complex128
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")


def create_form(
    form: GeneralForm,
    mesh: dolfinx.mesh.Mesh,
    coefficient_map: dict[str, dolfinx.fem.Function],
    constant_map: dict[str, dolfinx.fem.Constant],
):
    sd = form.ufl_form.subdomain_data()
    (domain,) = list(sd.keys())  # Assuming single domain

    # Subdomain markers (possibly empty list for some integral types)
    subdomains = {
        dolfinx.fem.forms._ufl_to_dolfinx_domain[key]: get_integration_domains(
            dolfinx.fem.forms._ufl_to_dolfinx_domain[key], subdomain_data[0]
        )
        for (key, subdomain_data) in sd.get(domain).items()
    }
    coefficients = {f"w{u.count()}": uh._cpp_object for (u, uh) in coefficient_map.items()}
    constants = {f"c{c.count()}": ch._cpp_object for (c, ch) in constant_map.items()}
    ftype = form_cpp_creator(dolfinx.default_scalar_type)
    f = ftype(
        form.module.ffi.cast("uintptr_t", form.module.ffi.addressof(form.ufcx_form)),
        [],
        coefficients,
        constants,
        subdomains,
        mesh._cpp_object,
    )
    return dolfinx.fem.Form(f, form.ufcx_form, form.code)


def test_compiled_form():
    # Create ufl form exclusively with basix.ufl and UFL
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
    compiled_form = compile_form(MPI.COMM_WORLD, J)

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
        form = create_form(compiled_form, mesh, {u: uh, w: wh}, {c: ch, e: eh})
        assert np.isclose(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM), 1.5)

    # Create various meshes, that all uses this compiled form with a map from ufl to dolfinx functions and constants
    for i in range(1, 4):
        create_and_integrate(i, compiled_form)
