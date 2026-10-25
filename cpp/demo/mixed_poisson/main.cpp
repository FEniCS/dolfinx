// # Mixed Poisson equation
//
// This demo illustrates how to solve the Poisson equation using a mixed
// (two-field) formulation. In particular, it illustrates how to:
//
// * Create a mixed finite element problem
// * Extract subspaces
// * Apply boundary conditions to different fields in a mixed problem
// * Create integration domain data to execute finite element kernels
//   over subsets of the boundary
// * Use a submesh to represent boundary data
//
// ## Equation and problem definition
//
// A mixed formulation of the Poisson equation is obtained by introducing
// the gradient $\sigma = \nabla u$ as an additional unknown. The
// equations then read
//
// \begin{align*}
//   \sigma - \nabla u &= 0 \quad {\rm in} \ \Omega, \\
//   \nabla \cdot \sigma &= - f \quad {\rm in} \ \Omega,
// \end{align*}
//
// with boundary conditions
//
// \begin{align*}
//   u &= u_0 \quad {\rm on} \ \Gamma_{D}, \\
//   \sigma \cdot n &= g \quad {\rm on} \ \Gamma_{N},
// \end{align*}
//
// where $n$ is the outward unit normal to the boundary. In this
// formulation the flux condition $\sigma \cdot n = g$ is essential (it
// is imposed on the function space), while $u = u_0$ is natural (it
// appears in the variational form). The variational problem reads: find
// $(\sigma, u) \in \Sigma_g \times V$ such that
//
// $$
//   a((\sigma, u), (\tau, v)) = L((\tau, v))
//   \quad \forall \ (\tau, v) \in \Sigma_0 \times V,
// $$
//
// where
//
// \begin{align*}
//   a((\sigma, u), (\tau, v)) &=
//     \int_{\Omega} \sigma \cdot \tau + \nabla \cdot \tau \ u
//   + \nabla \cdot \sigma \ v \ {\rm d} x, \\
//   L((\tau, v)) &= - \int_{\Omega} f v \ {\rm d} x
//   + \int_{\Gamma_D} u_0 \tau \cdot n \ {\rm d} s,
// \end{align*}
//
// $\Sigma_g = \{ \tau \in H({\rm div}, \Omega) : \tau \cdot n = g \
// {\rm on} \ \Gamma_N \}$ and $V = L^2(\Omega)$.
//
// To discretise, we choose finite-dimensional subspaces
// $\Sigma_{h} \subset H({\rm div}, \Omega)$ and $V_h \subset V$. A
// stable choice is Raviart-Thomas elements of degree $k$ for
// $\Sigma_{h}$ and discontinuous Lagrange elements of degree $k-1$ for
// $V_h$. This demo uses $k = 1$.
//
// The boundaries are defined as in the
// {doc}`Poisson demo <demo_poisson>`, but on the unit square:
//
// * $\Omega = [0,1] \times [0,1]$
// * $\Gamma_{D} = \{(x, y) \in \partial \Omega : x = 0 \ {\rm or} \ x = 1\}$
// * $\Gamma_{N} = \partial \Omega \setminus \Gamma_{D}$
// * $u_0 = 20 y + 1$
// * $g = 10$
// * $f = \sin(5x) + 1$
//
// The boundary data $u_0$ is represented by a function on a submesh of
// the facets of $\Gamma_{D}$. Assembling $L$ therefore involves two
// meshes: the integration domain (the full mesh) and the submesh.
//
// ## Implementation
//
// Running this demo requires the files:
// {download}`demo_mixed_poisson/main.cpp`,
// {download}`demo_mixed_poisson/mixed_poisson.py` and
// {download}`demo_mixed_poisson/CMakeLists.txt`.
//
// ### UFL form file
//
// The UFL forms are defined in
// {download}`demo_mixed_poisson/mixed_poisson.py`.
// ````{admonition} UFL form implemented in Python
// :class: dropdown
// ![ufl-code]
// ````
//
// ### C++ program
//
// The `main` function performs the following steps:
//
// 1. Create the mesh, the mixed Raviart-Thomas/discontinuous Lagrange
//    space and its subspaces.
// 2. Interpolate $f$, and interpolate a vector field whose normal
//    component on $\Gamma_{N}$ is $g$.
// 3. Create a submesh of the facets of $\Gamma_{D}$ with
//    {cpp:func}`dolfinx::mesh::create_submesh`, and interpolate $u_0$
//    into a Lagrange space on it.
// 4. Create a {cpp:class}`dolfinx::fem::DirichletBC` that constrains the
//    normal component of $\sigma$ on $\Gamma_{N}$.
// 5. Create integration domain data for the facets of $\Gamma_{D}$, which
//    carry the `ds(1)` integral in the UFL file.
// 6. Create the forms, passing the
//    {cpp:class}`dolfinx::mesh::EntityMap` that relates the submesh to
//    the mesh, then assemble and solve the linear system.

#include "mixed_poisson.h"
#include <basix/cell.h>
#include <basix/finite-element.h>
#include <basix/mdspan.hpp>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/petsc.h>
#include <map>
#include <memory>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_t<T>;

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  common::petsc::check(PetscInitialize(&argc, &argv, nullptr, nullptr),
                       "PetscInitialize");

  {
    mesh::CellType cell_type = mesh::CellType::triangle;

    // Create mesh
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_rectangle<U>(
        MPI_COMM_WORLD, {{{0, 0}, {1, 1}}}, {32, 32}, cell_type));

    // Create Basix elements
    basix::cell::type basix_cell_type
        = dolfinx::mesh::cell_type_to_basix_type(cell_type);
    auto RT
        = basix::create_element<U>(basix::element::family::RT, basix_cell_type,
                                   1, basix::element::lagrange_variant::unset,
                                   basix::element::dpc_variant::unset, false);
    auto P0
        = basix::create_element<U>(basix::element::family::P, basix_cell_type,
                                   0, basix::element::lagrange_variant::unset,
                                   basix::element::dpc_variant::unset, true);

    // Create DOLFINx mixed element
    auto ME = std::make_shared<fem::FiniteElement<U>>(
        std::vector<fem::BasixElementData<U>>{{RT}, {P0}},
        mesh->geometry().dim());

    // Create FunctionSpace
    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace<U>(mesh, ME));

    // Get subspaces (views into V)
    auto V0 = std::make_shared<fem::FunctionSpace<U>>(V->sub({0}));
    auto V1 = std::make_shared<fem::FunctionSpace<U>>(V->sub({1}));

    // Collapse spaces
    auto W0 = std::make_shared<fem::FunctionSpace<U>>(V0->collapse().first);
    auto W1 = std::make_shared<fem::FunctionSpace<U>>(V1->collapse().first);

    // Create source function and interpolate sin(5x) + 1
    auto f = std::make_shared<fem::Function<T>>(W1);
    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f(x.extent(1));
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto x0 = x(0, p);
            f[p] = std::sin(5 * x0) + 1;
          }
          return {std::move(f), std::vector<std::size_t>{f.size()}};
        });

    // Boundary value for sigma, chosen so that sigma.n = 10 on the top
    // and bottom boundaries
    auto g = std::make_shared<fem::Function<T>>(W0);
    g->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          using mspan_t
              = md::mdspan<T, md::extents<std::size_t, 2, md::dynamic_extent>>;

          std::vector<T> fdata(2 * x.extent(1), 0);
          mspan_t f(fdata.data(), 2, x.extent(1));
          for (std::size_t p = 0; p < x.extent(1); ++p)
            f(1, p) = x(1, p) < 0.5 ? -10 : 10;
          return {std::move(fdata), std::vector<std::size_t>{2, x.extent(1)}};
        });

    // Get list of all boundary facets
    mesh->topology()->create_connectivity(1, 2);
    std::vector bfacets = mesh::exterior_facet_indices(*mesh->topology());

    // Get facets with boundary condition on u
    std::vector<std::int32_t> dfacets = mesh::locate_entities_boundary(
        *mesh, 1,
        [](auto x)
        {
          using coord_t = typename decltype(x)::value_type;
          constexpr coord_t eps = 1e-8;
          std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto x0 = x(0, p);
            if (std::abs(x0) < eps or std::abs(x0 - 1) < eps)
              marker[p] = true;
          }
          return marker;
        });

    // Create a submesh of the facets in dfacets on which to represent
    // u0. The returned EntityMap relates submesh entities to mesh
    // entities, and is needed to assemble the mixed-domain form.
    int tdim = mesh->topology()->dim();
    int fdim = tdim - 1;

    auto submesh_data = [](auto& mesh, int dim, auto&& dfacets)
    {
      auto [submesh, e_map, v_map, g_map]
          = mesh::create_submesh(mesh, dim, dfacets);
      return std::pair(std::make_shared<mesh::Mesh<U>>(std::move(submesh)),
                       std::move(e_map));
    };
    auto [submesh, entity_map] = submesh_data(*mesh, fdim, dfacets);

    // Create an element for u0
    basix::cell::type submesh_cell_type
        = dolfinx::mesh::cell_type_to_basix_type(
            submesh->topology()->cell_type());

    auto Qe = std::make_shared<fem::FiniteElement<U>>(
        basix::create_element<U>(basix::element::family::P, submesh_cell_type,
                                 1, basix::element::lagrange_variant::unset,
                                 basix::element::dpc_variant::unset, false),
        submesh->geometry().dim());

    // Create a function space for u0 on the submesh
    auto Q = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace<U>(submesh, Qe));

    // Interpolate the boundary value u0 = 20y + 1
    auto u0 = std::make_shared<fem::Function<T>>(Q);
    u0->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f(x.extent(1));
          for (std::size_t p = 0; p < x.extent(1); ++p)
            f[p] = 20 * x(1, p) + 1;
          return {std::move(f), std::vector<std::size_t>{f.size()}};
        });

    // Write u0 to file to visualise
    io::VTKFile u0_file(MPI_COMM_WORLD, "u0.pvd", "w");
    u0_file.write<T>({*u0}, 0);

    // Facets with the flux (sigma) boundary condition: all boundary
    // facets except those in dfacets
    std::vector<std::int32_t> nfacets;
    std::ranges::set_difference(bfacets, dfacets, std::back_inserter(nfacets));

    // Get the sigma degrees-of-freedom on these facets, in V and in the
    // collapsed space W0
    std::array<std::vector<std::int32_t>, 2> ndofs
        = fem::locate_dofs_topological(
            *mesh->topology(), {*V0->dofmap(), *W0->dofmap()}, 1, nfacets);

    // Create the boundary condition, constraining sigma.n to the normal
    // component of g. V0 is the constrained subspace.
    fem::DirichletBC<T> bc(g, ndofs, V0);

    // Compute the integration entities for the facets in dfacets, on
    // which the ds(1) integral in the UFL file is evaluated
    std::vector<std::int32_t> domains = fem::compute_integration_domains(
        fem::IntegralType::exterior_facet, *mesh->topology(), dfacets);

    // Map exterior facet integrals with subdomain ID 1 to these
    // integration entities
    std::map<
        fem::IntegralType,
        std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>
        subdomain_data{{fem::IntegralType::exterior_facet, {{1, domains}}}};

    // Define the bilinear form
    fem::Form<T> a = fem::create_form<T>(*form_mixed_poisson_a, {V, V}, {}, {},
                                         subdomain_data, {});

    // L involves mesh (the integration domain, passed last) and submesh
    // (on which u0 is defined). An EntityMap must be supplied for each
    // mesh other than the integration domain.
    fem::Form<T> L = fem::create_form<T>(
        *form_mixed_poisson_L, {V}, {{"f", f}, {"u0", u0}}, {}, subdomain_data,
        {entity_map}, V->mesh());

    // Create solution finite element Function
    auto u = std::make_shared<fem::Function<T>>(V);

    // Create matrix and RHS vector data structures
    auto A = la::petsc::Matrix(fem::petsc::create_matrix(a), false);
    la::Vector<T> b(L.function_spaces()[0]->dofmap()->index_map,
                    L.function_spaces()[0]->dofmap()->index_map_bs());

    // Assemble the bilinear form into a matrix. Rows and columns of
    // Dirichlet dofs are zeroed.
    common::petsc::check(MatZeroEntries(A.mat()), "MatZeroEntries");
    fem::assemble_matrix(la::petsc::Matrix::set_fn(A.mat(), ADD_VALUES), a,
                         {bc});

    // Set '1' on diagonal for Dirichlet dofs. Their rows were zeroed by
    // assembly, so the value is added, which avoids flushing the matrix
    // to switch from adding to inserting.
    fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A.mat(), ADD_VALUES), *V,
                         {bc});
    common::petsc::check(MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY),
                         "MatAssemblyBegin");
    common::petsc::check(MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY),
                         "MatAssemblyEnd");

    // Assemble the linear form `L` into RHS vector
    std::ranges::fill(b.array(), 0);
    fem::assemble_vector(b.array(), L);

    // Modify unconstrained dofs on RHS to account for Dirichlet BC dofs
    // (constrained dofs), and perform parallel update on the vector.
    fem::apply_lifting(b.array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());

    // Set value for constrained dofs
    bc.set(b.array(), std::nullopt);

    // Create PETSc linear solver
    la::petsc::KrylovSolver lu(MPI_COMM_WORLD);
    common::petsc::set_option("ksp_type", "preonly");
    common::petsc::set_option("pc_type", "lu");
    if (sizeof(PetscInt) == 4)
      common::petsc::set_option("pc_factor_mat_solver_type", "mumps");
    else
      common::petsc::set_option("pc_factor_mat_solver_type", "superlu_dist");
    lu.set_from_options();

    // Solve linear system Ax = b
    lu.set_operator(A.mat());
    la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
    la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);
    if (lu.solve(_u.vec(), _b.vec()) < 0)
      throw std::runtime_error("Linear solver did not converge.");

    // The KSP object is available for anything the solver does not
    // wrap, here the number of linear solver iterations
    PetscInt num_it = 0;
    common::petsc::check(KSPGetIterationNumber(lu.ksp(), &num_it),
                         "KSPGetIterationNumber");
    std::cout << "Number of linear solver iterations: " << num_it << std::endl;

    // Update ghost values before output
    u->x()->scatter_fwd();

    // Save solution in VTK format
    auto u_soln = std::make_shared<fem::Function<T>>(u->sub(1).collapse());
    io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    file.write<T>({*u_soln}, 0);

#ifdef HAS_ADIOS2
    // Save solution in VTX format
    io::VTXWriter<U> vtx_u(MPI_COMM_WORLD, "u.bp", {u_soln}, "bp4");
    vtx_u.write(0);
    // Save interpolated boundary condition
    io::VTXWriter<U> vtx_u0(MPI_COMM_WORLD, "u0.bp", {u0}, "bp4");
    vtx_u0.write(0);
#endif
  }

  common::petsc::check(PetscFinalize(), "PetscFinalize");

  return 0;
}
