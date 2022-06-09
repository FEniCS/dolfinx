from dolfinx import io, fem, plot, cpp
from petsc4py import PETSc
import ufl
from ufl import FacetNormal, as_vector, inner, grad, curl, cross, lhs, rhs, dot, conj, sqrt
from datetime import datetime
import numpy as np
import gmsh
from mpi4py import MPI
import os
import sys
from gmsh_helpers import gmsh_model_to_mesh
from utils import background_electric_field, curl_2d, radial_distance, from_2d_to_3d, calculateAnalyticalEfficiencies, save_as_xdmf
from scipy.constants import mu_0, epsilon_0

# constant definition
um = 10**-6 #micron
nm = 10**-9 #nanometer
pi = np.pi 

# radius of the wire and of the boundary of the domain
radius_wire = 0.050*um
radius_dom = 1*um

# the smaller the mesh_factor, the finer is the mesh
mesh_factor = 0.5

# finite element degree
degree = 3

# wavelength sweep
wl0 = 0.4*um

# background refractive index
n_bkg = 1.33
eps_bkg = n_bkg**2

# mesh size inside the wire
in_wire_size = mesh_factor*7*nm

# mesh size at the boundary of the wire
on_wire_size = mesh_factor*3*nm

# mesh size in the vacuum
bkg_size = mesh_factor*60*nm

# mesh size at the boundary
boundary_size = mesh_factor*30*nm

# tags for the subdomains
au_tag = 1          # gold wire
bkg_tag = 2         # background
boundary_tag = 3    # boundary

# mesh definition in gmsh

gmsh.initialize(sys.argv)
if MPI.COMM_WORLD.rank == 0:
   
    gmsh.model.add("nanowire")

    # a dummy boundary is added for setting a finer mesh
    gmsh.model.occ.addCircle(0, 0, 0, radius_wire*0.8, angle1 = 0, angle2 = 2*pi, tag=1)
    gmsh.model.occ.addCircle(0, 0, 0, radius_wire, angle1 = 0, angle2 = 2*pi, tag=2)
    
    # a dummy boundary is added for setting a finer mesh
    gmsh.model.occ.addCircle(0, 0, 0, radius_dom*0.9, angle1 = 0, angle2 = 2*pi, tag=3)
    gmsh.model.occ.addCircle(0, 0, 0, radius_dom, angle1 = 0, angle2 = 2*pi, tag=4)

    gmsh.model.occ.addCurveLoop([1], tag=1)
    gmsh.model.occ.addPlaneSurface([1], tag=1)

    gmsh.model.occ.addCurveLoop([2], tag=2)
    gmsh.model.occ.addCurveLoop([1], tag=3)
    gmsh.model.occ.addPlaneSurface([2, 3], tag=2)

    gmsh.model.occ.addCurveLoop([3], tag=4)
    gmsh.model.occ.addCurveLoop([2], tag=5)
    gmsh.model.occ.addPlaneSurface([4, 5], tag=3)

    gmsh.model.occ.addCurveLoop([4], tag=6)
    gmsh.model.occ.addCurveLoop([3], tag=7)
    gmsh.model.occ.addPlaneSurface([6, 7], tag=4)

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [1, 2], tag=au_tag) 
    gmsh.model.addPhysicalGroup(2, [3, 4], tag=bkg_tag) 

    gmsh.model.addPhysicalGroup(1, [4], tag=boundary_tag) 

    gmsh.model.mesh.setSize([(0, 1)], size = in_wire_size)
    gmsh.model.mesh.setSize([(0, 2)], size = on_wire_size)
    gmsh.model.mesh.setSize([(0, 3)], size = bkg_size)
    gmsh.model.mesh.setSize([(0, 4)], size = boundary_size)

    gmsh.model.mesh.generate(2)

    mesh, cell_tags, facet_tags = gmsh_model_to_mesh(gmsh.model, cell_data=True, facet_data=True, gdim=2)

MPI.COMM_WORLD.barrier()

# definition of finite element for the electric field
curl_el = ufl.FiniteElement("N1curl", mesh.ufl_cell(), 3)

# definition of finite element for the r function (see next)
lagr_el = ufl.FiniteElement("CG", mesh.ufl_cell(), 2)

# function space for the electric field
V = fem.FunctionSpace(mesh, curl_el)

# wavevector of the background field
k0 = 2*np.pi/wl0

# angle of incidence of the background field with respect to the horizontal axis
deg = np.pi/180
theta = 45*deg

# plane wave function
f = background_electric_field(theta, n_bkg, k0)
Eb = fem.Function(V, dtype = np.complex128)
Eb.interpolate(f.eval)

# function r = radial distance from the (0, 0) point
lagr_space = fem.FunctionSpace(mesh, lagr_el)
r = fem.Function(lagr_space, dtype = np.complex128)
r.interpolate(radial_distance)

# definition of Trial and Test functions
Es = ufl.TrialFunction(V)
Vs = ufl.TestFunction(V)

# definition of 3d fields for cross and curl operations
Es_3d = from_2d_to_3d(Es)
Vs_3d = from_2d_to_3d(Vs)

# Measures for subdomains
dAu = ufl.Measure("dx", mesh, subdomain_data=cell_tags, subdomain_id=au_tag)
dBkg = ufl.Measure("dx", mesh, subdomain_data=cell_tags, subdomain_id=bkg_tag)
dsbc = ufl.Measure("ds", mesh, subdomain_data=facet_tags, subdomain_id=boundary_tag)
dDom = dAu + dBkg

# normal to the boundary
n = FacetNormal(mesh)
n_3d = from_2d_to_3d(n)

# definition of relative permittivity for Au @400nm
reps_au = -1.0782
ieps_au = 5.8089
eps_au = reps_au + ieps_au*1j

# definition of the relative permittivity over the whole domain
D = fem.FunctionSpace(mesh, ("DG", 0))
eps = fem.Function(D)
au_cells = cell_tags.indices[cell_tags.values==au_tag]
bkg_cells = cell_tags.indices[cell_tags.values==bkg_tag]
eps.x.array[au_cells] = np.full(len(au_cells), reps_au + ieps_au*1j)
eps.x.array[bkg_cells] = np.full(len(bkg_cells), eps_bkg)

# weak form
F = - inner(curl(Es), curl(Vs))*dDom \
    + eps*k0**2*inner(Es, Vs)*dDom \
    + k0**2*(eps-eps_bkg)*inner(Eb, Vs)*dDom \
    + (1j*k0*n_bkg + 1/(2*r))*inner(cross(Es_3d, n_3d), cross(Vs_3d, n_3d))*dsbc # scattering boundary condition

# splitting in left-hand side and right-hand side
a, L = lhs(F), rhs(F)

problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
Eh = problem.solve()

# total electric field E = Es + Eb
E = fem.Function(V, dtype=np.complex128)
E.x.array[:] = Eb.x.array[:] + Eh.x.array[:]

# ||E||
norm_func = sqrt(inner(Eh, Eh))
V_normEh = fem.FunctionSpace(mesh, lagr_el)
norm_expr = fem.Expression(norm_func, V_normEh.element.interpolation_points)
normEh = fem.Function(V_normEh)
normEh.interpolate(norm_expr)

# save the fields as xdmf files
save_as_xdmf("data/Es.xdmf", mesh, Eh)
save_as_xdmf("data/E.xdmf", mesh, E)
save_as_xdmf("data/normEh.xdmf", mesh, normEh)

# calculation of analytical efficiencies
q_abs_analyt, q_sca_analyt, q_ext_analyt = calculateAnalyticalEfficiencies(reps_au, ieps_au, n_bkg, wl0, radius_wire) 

# vacuum impedance
Z0 = np.sqrt(mu_0/epsilon_0)

# magnetic field H
Hh_3d = -1j*curl_2d(Eh)/Z0/k0/n_bkg

Eh_3d = from_2d_to_3d(Eh)
E_3d = from_2d_to_3d(E)

# intensity of the electromagnetic fields I0 = 0.5*E0**2/Z0 
# E0 = np.sqrt(ax**2 + ay**2) = 1, see background_electric_field
I0 = 0.5/Z0

# geometrical cross section of the wire
gcs = 2*radius_wire

# quantities for the calculation of efficiencies
P = 0.5*inner(cross(Eh_3d,conj(Hh_3d)),n_3d)
Q = 0.5*ieps_au*k0*(inner(E_3d,E_3d))/Z0/n_bkg

# normalized efficiencies
q_abs_fenics = ufl.real(fem.assemble_scalar(fem.form(Q*dAu))/gcs/I0)
q_sca_fenics = ufl.real(fem.assemble_scalar(fem.form(P*dsbc))/gcs/I0)
q_ext_fenics = q_abs_fenics + q_sca_fenics

err_abs = np.abs(q_abs_analyt - q_abs_fenics)/q_abs_analyt*100
err_sca = np.abs(q_sca_analyt - q_sca_fenics)/q_sca_analyt*100
err_ext = np.abs(q_ext_analyt - q_ext_fenics)/q_ext_analyt*100

print()
print(f"The analytical absorption efficiency is {q_abs_analyt}")
print(f"The numerical absorption efficiency is {q_abs_fenics}")
print(f"The error is {err_abs}%")
print()
print(f"The analytical scattering efficiency is {q_sca_analyt}")
print(f"The numerical scattering efficiency is {q_sca_fenics}")
print(f"The error is {err_sca}%")
print()
print(f"The analytical extinction efficiency is {q_ext_analyt}")
print(f"The numerical extinction efficiency is {q_ext_fenics}")
print(f"The error is {err_ext}%")


