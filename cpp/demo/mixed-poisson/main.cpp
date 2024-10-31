
#include "poisson.h"
#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/petsc.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
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
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}},
                                  {32, 16}, mesh::CellType::triangle, part));

    auto RT_b = basix::create_element<U>(
        basix::element::family::RT, basix::cell::type::triangle, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);
    auto P0_b = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::triangle, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, true);

    auto RT = std::make_shared<fem::FiniteElement<U>>(RT_b, 1);
    auto P0 = std::make_shared<fem::FiniteElement<U>>(P0_b, 1);

    // Create element
    std::vector<std::shared_ptr<const fem::FiniteElement<U>>> _ME = {RT, P0};
    auto ME = std::make_shared<fem::FiniteElement<U>>(_ME);

    std::function<void(std::span<std::int32_t>, std::uint32_t)> permute_inv
        = nullptr;
    if (ME->needs_dof_permutations())
      permute_inv = ME->dof_permutation_fn(true, true);

    // // Create dofmap
    fem::ElementDofLayout layout = fem::create_element_dof_layout(*ME);
    auto dofmap = std::make_shared<fem::DofMap>(fem::create_dofmap(
        MPI_COMM_WORLD, layout, *mesh->topology(), permute_inv, nullptr));

    // std::vector<
    std::vector<std::size_t> vs = {3};
    auto V = std::make_shared<fem::FunctionSpace<U>>(mesh, ME, dofmap, vs);
    // auto V = std::make_shared<fem::FunctionSpace<U>>(
    //     fem::create_functionspace<U>(mesh, ME, vs));

    // const std::vector<std::shared_ptr<const FiniteElement<geometry_type>>>&
    //     elements);

    // std::vector<std::shared_ptr<fem::FiniteElement<U>>> elements;

    // auto P0 = std::make_shared<basix::create_element<U>>(
    //     basix::element::family::P, basix::cell::type::triangle, 0,
    //     basix::element::lagrange_variant::unset,
    //     basix::element::dpc_variant::unset, true);

    // std::vector<std::shared_ptr<fem::FiniteElement<U>>> elements;
    // elements.push_back(RT);
    // = {RT, P0};
    // fem::FiniteElement<U> ME({RT, P0});
    // const std::vector<std::shared_ptr<const FiniteElement<geometry_type>>>&
    //     elements);

    // auto V = std::make_shared<fem::FunctionSpace<U>>(
    //     fem::create_functionspace(mesh, ME, {}));

    //     //  Next, we define the variational formulation by initializing the
    //     //  bilinear and linear forms ($a$, $L$) using the previously
    //     //  defined {cpp:class}`FunctionSpace` `V`.  Then we can create the
    //     //  source and boundary flux term ($f$, $g$) and attach these to the
    //     //  linear form.

    //     // Prepare and set Constants for the bilinear form
    //     auto kappa = std::make_shared<fem::Constant<T>>(2.0);
    //     auto f = std::make_shared<fem::Function<T>>(V);
    //     auto g = std::make_shared<fem::Function<T>>(V);

    //     // Define variational forms
    //     auto a = std::make_shared<fem::Form<T>>(fem::create_form<T>(
    //         *form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}, {}));
    //     auto L = std::make_shared<fem::Form<T>>(fem::create_form<T>(
    //         *form_poisson_L, {V}, {{"f", f}, {"g", g}}, {}, {}, {}));

    //     //  Now, the Dirichlet boundary condition ($u = 0$) can be created
    //     //  using the class {cpp:class}`DirichletBC`. A
    //     //  {cpp:class}`DirichletBC` takes two arguments: the value of the
    //     //  boundary condition, and the part of the boundary on which the
    //     //  condition applies. In our example, the value of the boundary
    //     //  condition (0.0) can represented using a {cpp:class}`Function`,
    //     //  and the Dirichlet boundary is defined by the indices of degrees
    //     //  of freedom to which the boundary condition applies. The
    //     //  definition of the Dirichlet boundary condition then looks as
    //     //  follows:

    //     // Define boundary condition

    //     auto facets = mesh::locate_entities_boundary(
    //         *mesh, 1,
    //         [](auto x)
    //         {
    //           using U = typename decltype(x)::value_type;
    //           constexpr U eps = 1.0e-8;
    //           std::vector<std::int8_t> marker(x.extent(1), false);
    //           for (std::size_t p = 0; p < x.extent(1); ++p)
    //           {
    //             auto x0 = x(0, p);
    //             if (std::abs(x0) < eps or std::abs(x0 - 2) < eps)
    //               marker[p] = true;
    //           }
    //           return marker;
    //         });
    //     const auto bdofs = fem::locate_dofs_topological(
    //         *V->mesh()->topology_mutable(), *V->dofmap(), 1, facets);
    //     auto bc = std::make_shared<const fem::DirichletBC<T>>(0.0, bdofs, V);

    //     f->interpolate(
    //         [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
    //         {
    //           std::vector<T> f;
    //           for (std::size_t p = 0; p < x.extent(1); ++p)
    //           {
    //             auto dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
    //             auto dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
    //             f.push_back(10 * std::exp(-(dx + dy) / 0.02));
    //           }

    //           return {f, {f.size()}};
    //         });

    //     g->interpolate(
    //         [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
    //         {
    //           std::vector<T> f;
    //           for (std::size_t p = 0; p < x.extent(1); ++p)
    //             f.push_back(std::sin(5 * x(0, p)));
    //           return {f, {f.size()}};
    //         });

    //     //  Now, we have specified the variational forms and can consider
    //     //  the solution of the variational problem. First, we need to
    //     //  define a {cpp:class}`Function` `u` to store the solution. (Upon
    //     //  initialization, it is simply set to the zero function.) Next, we
    //     //  can call the `solve` function with the arguments `a == L`, `u`
    //     //  and `bc` as follows:

    //     auto u = std::make_shared<fem::Function<T>>(V);
    //     auto A = la::petsc::Matrix(fem::petsc::create_matrix(*a), false);
    //     la::Vector<T> b(L->function_spaces()[0]->dofmap()->index_map,
    //                     L->function_spaces()[0]->dofmap()->index_map_bs());

    //     MatZeroEntries(A.mat());
    //     fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(),
    //     ADD_VALUES),
    //                          *a, {bc});
    //     MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
    //     MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
    //     fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A.mat(),
    //     INSERT_VALUES), *V,
    //                          {bc});
    //     MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    //     MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

    //     b.set(0.0);
    //     fem::assemble_vector(b.mutable_array(), *L);
    //     fem::apply_lifting<T, U>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    //     b.scatter_rev(std::plus<T>());
    //     bc->set(b.mutable_array(), std::nullopt);

    //     la::petsc::KrylovSolver lu(MPI_COMM_WORLD);
    //     la::petsc::options::set("ksp_type", "preonly");
    //     la::petsc::options::set("pc_type", "lu");
    //     lu.set_from_options();

    //     lu.set_operator(A.mat());
    //     la::petsc::Vector _u(la::petsc::create_vector_wrap(*u->x()), false);
    //     la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);
    //     lu.solve(_u.vec(), _b.vec());

    //     // Update ghost values before output
    //     u->x()->scatter_fwd();

    //     //  The function `u` will be modified during the call to solve. A
    //     //  {cpp:class}`Function` can be saved to a file. Here, we output
    //     //  the solution to a `VTK` file (specified using the suffix `.pvd`)
    //     //  for visualisation in an external program such as Paraview.

    //     // Save solution in VTK format
    //     io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    //     file.write<T>({*u}, 0.0);

    // #ifdef HAS_ADIOS2
    //     // Save solution in VTX format
    //     io::VTXWriter<U> vtx(MPI_COMM_WORLD, "u.bp", {u}, "bp4");
    //     vtx.write(0);
    // #endif
  }

  PetscFinalize();

  return 0;
}
