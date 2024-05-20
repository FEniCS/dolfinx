#include "poisson.h"
#include "superlu.h"

#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/Vector.h>
#include <utility>
#include <vector>

using namespace dolfinx;
using T = SCALAR;
using U = double;

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);

  spdlog::set_level(spdlog::level::info);

  {
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_box<U>(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {32, 32, 320},
        mesh::CellType::tetrahedron, part));

    int tdim = mesh->topology()->dim();
    mesh->topology()->create_entities(tdim - 1);

    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::tetrahedron, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(mesh, element, {}));

    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<fem::Constant<T>>(2.0);
    auto f = std::make_shared<fem::Function<T, U>>(V);
    auto g = std::make_shared<fem::Function<T, U>>(V);

    // Define variational forms
    auto a = std::make_shared<fem::Form<T, U>>(fem::create_form<T, U>(
        *form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}));
    auto L = std::make_shared<fem::Form<T, U>>(fem::create_form<T, U>(
        *form_poisson_L, {V}, {{"f", f}, {"g", g}}, {}, {}));

    // Define boundary condition

    auto facets = mesh::locate_entities_boundary(
        *mesh, tdim - 1,
        [](auto x)
        {
          using U = typename decltype(x)::value_type;
          constexpr U eps = 1.0e-8;
          std::vector<std::int8_t> marker(x.extent(1), false);
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto x0 = x(0, p);
            if (std::abs(x0) < eps or std::abs(x0 - 2) < eps)
              marker[p] = true;
          }
          return marker;
        });
    const auto bdofs = fem::locate_dofs_topological(
        *V->mesh()->topology_mutable(), *V->dofmap(), tdim - 1, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T, U>>(0.0, bdofs, V);

    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
            auto dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
            f.push_back(10 * std::exp(-(dx + dy) / 0.02));
          }

          return {f, {f.size()}};
        });

    g->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> f;
          for (std::size_t p = 0; p < x.extent(1); ++p)
            f.push_back(std::sin(5 * x(0, p)));
          return {f, {f.size()}};
        });

    // Compute solution
    fem::Function<T, U> u(V);
    la::SparsityPattern sp = fem::create_sparsity_pattern(*a);
    sp.finalize();
    la::MatrixCSR<T> A(sp);
    la::Vector<T> b(L->function_spaces()[0]->dofmap()->index_map,
                    L->function_spaces()[0]->dofmap()->index_map_bs());

    A.set(0.0);
    fem::assemble_matrix(A.mat_add_values(), *a, {bc});
    A.scatter_rev();
    fem::set_diagonal<T, U>(A.mat_set_values(), *V, {bc});

    b.set(0.0);
    fem::assemble_vector(b.mutable_array(), *L);
    fem::apply_lifting<T, U>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    b.scatter_rev(std::plus<T>());
    fem::set_bc<T, U>(b.mutable_array(), {bc});

    // Solver: A.u = b
    // dolfinx::common::Timer tfac("[MUMPS Factorize]");
    // superlu_Solver<T> LU(mesh->comm());
    // LU.set_operator(A);
    // tfac.stop();

    dolfinx::common::Timer tsolve("[SuperLU Solve]");
    superlu_solver(mesh->comm(), A, b, *u.x(), true);
    tsolve.stop();

    // Save solution in VTK format
    io::VTKFile file(MPI_COMM_WORLD, "u.pvd", "w");
    if constexpr (std::is_same_v<T, float>)
    {
      fem::Function<double, U> udouble(V);
      // Convert float to double
      std::copy(u.x()->array().begin(), u.x()->array().end(),
                udouble.x()->mutable_array().begin());
      file.write<double>({udouble}, 0.0);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>)
    {
      fem::Function<std::complex<double>, U> udouble(V);
      // Convert float to double
      std::copy(u.x()->array().begin(), u.x()->array().end(),
                udouble.x()->mutable_array().begin());
      file.write<std::complex<double>>({udouble}, 0.0);
    }
    else
      file.write<T>({u}, 0.0);
  }

  dolfinx::list_timings(MPI_COMM_WORLD, {TimingType::wall});

  MPI_Finalize();
  return 0;
}
