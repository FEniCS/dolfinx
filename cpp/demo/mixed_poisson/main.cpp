
#include "poisson.h"
#include <basix/finite-element.h>
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
using U = typename dolfinx::scalar_value_type_t<T>;

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {
    // Create mesh
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}},
                                  {46, 46}, mesh::CellType::triangle));

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
        std::vector<
            std::tuple<std::reference_wrapper<const basix::FiniteElement<U>>,
                       std::size_t, bool>>{{RT, 1, false}, {P0, 1, false}});

    // Create dof permutation function
    std::function<void(std::span<std::int32_t>, std::uint32_t)> permute_inv
        = nullptr;
    if (ME->needs_dof_permutations())
      permute_inv = ME->dof_permutation_fn(true, true);

    // Create dofmap
    fem::ElementDofLayout layout = fem::create_element_dof_layout(*ME);
    auto dofmap = std::make_shared<fem::DofMap>(fem::create_dofmap(
        MPI_COMM_WORLD, layout, *mesh->topology(), permute_inv, nullptr));

    // TODO: Allow mixed FunctionSpace to be created from DOLFINx
    // elements (not just Basix elements) via fem::create_functionspace.

    // Create Dofmap
    std::vector<std::size_t> vs = {3};
    auto V = std::make_shared<fem::FunctionSpace<U>>(mesh, ME, dofmap, vs);

    // Get subspaces (views into V)
    auto V0 = std::make_shared<fem::FunctionSpace<U>>(V->sub({0}));
    auto V1 = std::make_shared<fem::FunctionSpace<U>>(V->sub({1}));

    // Collapse spaces
    auto W0 = std::make_shared<fem::FunctionSpace<U>>(V0->collapse().first);
    auto W1 = std::make_shared<fem::FunctionSpace<U>>(V1->collapse().first);

    // TODO: Add Function that takes lambda function to interpolate?
    // Source function
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

    // Boundary condition value for u
    auto u0 = std::make_shared<fem::Function<T>>(W1);
    u0->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
            f.push_back(20 * x(1, p) + 1);
          return {f, {f.size()}};
        });

    // Get list of boundary facets
    mesh->topology()->create_connectivity(1, 2);
    std::vector bfacets = mesh::exterior_facet_indices(*mesh->topology());
    // TODO: are facet indices guaranteed to be sorted?
    if (!std::ranges::is_sorted(bfacets))
      throw std::runtime_error("Not sorted");

    // Get facets with boundary condition on u
    std::vector<std::int32_t> dfacets = locate_entities_boundary(
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
    // TODO: are facet indices guaranteed to be sorted?
    if (!std::ranges::is_sorted(dfacets))
      throw std::runtime_error("Not sorted");

    // Facets with \sigma (flux) boundary condition
    std::vector<std::int32_t> nfacets;
    std::ranges::set_difference(bfacets, dfacets, std::back_inserter(nfacets));
    if (dfacets.size() + nfacets.size() != bfacets.size())
      throw std::runtime_error("Inconsistent facets numbers.");

    // Get dofs that are constrained by a flux (\sigma)
    std::array<std::vector<std::int32_t>, 2> ndofs
        = fem::locate_dofs_topological(
            *mesh->topology(), {*V0->dofmap(), *W0->dofmap()}, 1, nfacets);

    // Create boundary condition for \sigma
    auto g = std::make_shared<fem::Function<T>>(W0);
    g->x()->set(10);
    fem::DirichletBC<T> bc(g, ndofs, V0);

    // Create integration domain for u boundary condition on ds(1)

    // Get facet data integration data for facets in dfacets
    std::vector<std::int32_t> domains = fem::compute_integration_domains(
        fem::IntegralType::exterior_facet, *mesh->topology(), dfacets);

    // Create data structure for ds(1) integration domain in form
    std::map<
        fem::IntegralType,
        std::vector<std::pair<std::int32_t, std::span<const std::int32_t>>>>
        subdomain_data = {{fem::IntegralType::exterior_facet, {{1, domains}}}};

    // Define variational forms and attach required data
    fem::Form<T> a = fem::create_form<T>(*form_poisson_a, {V, V}, {}, {},
                                         subdomain_data, {});
    fem::Form<T> L = fem::create_form<T>(
        *form_poisson_L, {V}, {{"f", f}, {"u0", u0}}, {}, subdomain_data, {});

    // Create solution Function
    auto u = std::make_shared<fem::Function<T>>(V);

    // Create matrix and RHS vector
    auto A = la::petsc::Matrix(fem::petsc::create_matrix(a), false);
    la::Vector<T> b(L.function_spaces()[0]->dofmap()->index_map,
                    L.function_spaces()[0]->dofmap()->index_map_bs());

    // Assemble matrix
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

    // Assemble RHS vector
    b.set(0.0);
    fem::assemble_vector(b.mutable_array(), L);

    // Modify unconstrained dofs on RHS to account for Dirichlet bcs
    fem::apply_lifting<T, U>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());

    // Set value for constrained dofs
    bc.set(b.mutable_array(), std::nullopt);

    // Create PETSc linear solver
    la::petsc::KrylovSolver lu(MPI_COMM_WORLD);
    la::petsc::options::set("ksp_type", "preonly");
    la::petsc::options::set("pc_type", "lu");
    la::petsc::options::set("pc_factor_mat_solver_type", "superlu");
    la::petsc::options::set("ksp_view");
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
