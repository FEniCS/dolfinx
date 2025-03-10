// # Mixed Poisson equation
//
// This demo illustrates how to solve Poisson equation using a mixed
// (two-field) formulation. In particular, it illustrates how to
//
// * Create a mixed finite element problem.
// * Extract subspaces.
// * Apply boundary conditions to different fields in a mixed problem.
// * Create integration domain data to execute finite element kernels.
//   over subsets of the boundary.
//
// The full implementation is in
// {download}`demo_mixed_poisson/main.cpp`.
//
//
// # Mixed formulation for the Poisson equation
//
// ## Equation and problem definition
//
// A mixed formulation of Poisson equation can be formulated by
// introducing an additional (vector) variable, namely the (negative)
// flux: $\sigma = \nabla u$. The partial differential equations
// then read
//
// $$
// \begin{align}
//   \sigma - \nabla u &= 0 \quad {\rm in} \ \Omega, \\
//   \nabla \cdot \sigma &= - f \quad {\rm in} \ \Omega,
// \end{align}
// $$
// with boundary conditions
//
// $$
//   u = u_0 \quad {\rm on} \ \Gamma_{D},  \\
//   \sigma \cdot n = g \quad {\rm on} \ \Gamma_{N}.
// $$
//
// where $n$ denotes the outward unit normal vector on the boundary. We
// see that the boundary condition for the flux ($\sigma \cdot n = g$)
// is an essential boundary condition (which should be enforced in
// the function space), while the other boundary condition ($u = u_0$)
// is a natural boundary condition (which should be applied to the
// variational form). Inserting the boundary conditions, this
// variational problem can be phrased in the general form: find
// $(\sigma, u) \in \Sigma_g \times V$ such that
//
// $$
//    a((\sigma, u), (\tau, v)) = L((\tau, v))
//    \quad \forall \ (\tau, v) \in \Sigma_0 \times V,
// $$
//
// where the forms $a$ and $L$ are defined as
//
// $$
// \begin{align}
//   a((\sigma, u), (\tau, v)) &:=
//     \int_{\Omega} \sigma \cdot \tau + \nabla \cdot \tau \ u
//   + \nabla \cdot \sigma \ v \ {\rm d} x, \\
//   L((\tau, v)) &:= - \int_{\Omega} f v \ {\rm d} x
//   + \int_{\Gamma_D} u_0 \tau \cdot n  \ {\rm d} s,
// \end{align}
// $$
// and $\Sigma_g := \{ \tau \in H({\rm div})$ such that $\tau \cdot
// n|_{\Gamma_N} = g \}$ and $V := L^2(\Omega)$.
//
// To discretize the above formulation, discrete function spaces
// $\Sigma_h \subset \Sigma$ and $V_h \subset V$ are needed to form a
// mixed function space $\Sigma_h \times V_h$. A stable choice of finite
// element spaces is to let $\Sigma_h$ be a Raviart-Thomas elements of
// polynomial order $k$ and $V_h$ be discontinuous elements of
// polynomial order $k-1$.
//
// We will use the same definitions of functions and boundaries as in the
// demo for {doc}`the Poisson equation <demo_poisson>`. These are:
//
// * $\Omega = [0,1] \times [0,1]$ (a unit square)
// * $\Gamma_{D} = \{(0, y) \cup (1, y) \in \partial \Omega\}$
// * $\Gamma_{N} = \{(x, 0) \cup (x, 1) \in \partial \Omega\}$
// * $u_0 = 20 y + 1$ on $\Gamma_{D}$
// * $g = 10$ (flux) on $\Gamma_{N}$
// * $f = \sin(5x - 0.5) + 1 (source term)

// ## UFL form file
//
// The UFL file is implemented in
// {download}`demo_mixed_poisson/mixed_poisson.py`.
// ````{admonition} UFL form implemented in python
// :class: dropdown
// ![ufl-code]
// ````
//

#include "mixed_poisson.h"
#include <basix/finite-element.h>
#include <basix/mdspan.hpp>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/petsc.h>
#include <map>
#include <memory>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_t<T>;

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {
    // Create mesh
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}},
                                  {32, 32}, mesh::CellType::triangle));

    // Create Basix elements
    auto RT = basix::create_element<U>(
        basix::element::family::RT, basix::cell::type::triangle, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);
    auto P0 = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::triangle, 0,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, true);

    // Create DOLFINx mixed element
    auto ME = std::make_shared<fem::FiniteElement<U>>(
        std::vector<fem::BasixElementData<U>>{{RT}, {P0}});

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
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto x0 = x(0, p);
            f.push_back(std::sin(5 * x0) + 1);
          }
          return {f, {f.size()}};
        });

    // Boundary condition value for u and interpolate 20y + 1
    auto u0 = std::make_shared<fem::Function<T>>(W1);
    u0->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
            f.push_back(20 * x(1, p) + 1);
          return {f, {f.size()}};
        });

    // Create boundary condition for \sigma and interpolate such that
    // flux = 10 (for top and bottom boundaries)
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
          return {std::move(fdata), {2, x.extent(1)}};
        });

    // Get list of all boundary facets
    mesh->topology()->create_connectivity(1, 2);
    std::vector bfacets = mesh::exterior_facet_indices(*mesh->topology());

    // Get facets with boundary condition on u
    std::vector<std::int32_t> dfacets = mesh::locate_entities_boundary(
        *mesh, 1,
        [](auto x)
        {
          using U = typename decltype(x)::value_type;
          constexpr U eps = 1.0e-8;
          std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto x0 = x(0, p);
            if (std::abs(x0) < eps or std::abs(x0 - 1) < eps)
              marker[p] = true;
          }
          return marker;
        });

    // Compute facets with \sigma (flux) boundary condition facets, which is
    // {all boundary facet} - {u0 boundary facets }
    std::vector<std::int32_t> nfacets;
    std::ranges::set_difference(bfacets, dfacets, std::back_inserter(nfacets));

    // Get dofs that are constrained by \sigma
    std::array<std::vector<std::int32_t>, 2> ndofs
        = fem::locate_dofs_topological(
            *mesh->topology(), {*V0->dofmap(), *W0->dofmap()}, 1, nfacets);

    // Create boundary condition for \sigma. \sigma \cdot n will be
    // constrained to to be equal to the normal component of g. The
    // boundary conditions are applied to degrees-of-freedom ndofs, and
    // V0 is the subspace that is constrained.
    fem::DirichletBC<T> bc(g, ndofs, V0);

    // Create integration domain data for u0 boundary condition (applied
    // on the ds(1) in the UFL file). First we get facet data
    // integration data for facets in dfacets.
    std::vector<std::int32_t> domains = fem::compute_integration_domains(
        fem::IntegralType::exterior_facet, *mesh->topology(), dfacets);

    // Create data structure for the ds(1) integration domain in form
    // (see the UFL file). It is for en exterior facet integral (the key
    // in the map), and exterior facet domain marked as '1' in the UFL
    // file, and 'domains' holds the necessary data to perform
    // integration of selected facets.
    std::map<
        fem::IntegralType,
        std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>
        subdomain_data{{fem::IntegralType::exterior_facet, {{1, domains}}}};

    // Define variational forms and attach he required data
    fem::Form<T> a = fem::create_form<T>(*form_mixed_poisson_a, {V, V}, {}, {},
                                         subdomain_data, {});
    fem::Form<T> L
        = fem::create_form<T>(*form_mixed_poisson_L, {V},
                              {{"f", f}, {"u0", u0}}, {}, subdomain_data, {});

    // Create solution finite element Function
    auto u = std::make_shared<fem::Function<T>>(V);

    // Create matrix and RHS vector data structures
    auto A = la::petsc::Matrix(fem::petsc::create_matrix(a), false);
    la::Vector<T> b(L.function_spaces()[0]->dofmap()->index_map,
                    L.function_spaces()[0]->dofmap()->index_map_bs());

    // Assemble the bilinear form into a matrix. The PETSc matrix is
    // 'flushed' so we can set values in it in the subsequent step.
    MatZeroEntries(A.mat());
    fem::assemble_matrix(la::petsc::Matrix::set_fn(A.mat(), ADD_VALUES), a,
                         {bc});
    MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);

    // Set '1' on diagonal for Dirichlet dofs
    fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A.mat(), INSERT_VALUES), *V,
                         {bc});
    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

    // Assemble the linear form L into RHS vector
    b.set(0);
    fem::assemble_vector(b.mutable_array(), L);

    // Modify unconstrained dofs on RHS to account for Dirichlet BC dofs
    // (constrained dofs), and perform parallel update on the vector.
    fem::apply_lifting<T, U>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());

    // Set value for constrained dofs
    bc.set(b.mutable_array(), std::nullopt);

    // Create PETSc linear solver
    la::petsc::KrylovSolver lu(MPI_COMM_WORLD);
    la::petsc::options::set("ksp_type", "preonly");
    la::petsc::options::set("pc_type", "lu");
    if (sizeof(PetscInt) == 4)
      la::petsc::options::set("pc_factor_mat_solver_type", "mumps");
    else
      la::petsc::options::set("pc_factor_mat_solver_type", "superlu_dist");
    lu.set_from_options();

    // Solve linear system Ax = b
    lu.set_operator(A.mat());
    la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
    la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);
    lu.solve(_u.vec(), _b.vec());

    // Update ghost values before output
    u->x()->scatter_fwd();

    // Save solution in VTK format
    auto u_soln = std::make_shared<fem::Function<T>>(u->sub(1).collapse());
    io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    file.write<T>({*u_soln}, 0);

#ifdef HAS_ADIOS2
    // Save solution in VTX format
    io::VTXWriter<U> vtx(MPI_COMM_WORLD, "u.bp", {u_soln}, "bp4");
    vtx.write(0);
#endif
  }

  PetscFinalize();

  return 0;
}
