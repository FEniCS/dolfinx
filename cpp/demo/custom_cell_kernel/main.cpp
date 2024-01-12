// Custom cell kernel (C++)
// .. code-block:: cpp

#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <utility>
#include <vector>

using namespace dolfinx;

using T = double;
using U = typename dolfinx::scalar_value_type_t<T>;

// .. code-block:: cpp

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    // Create mesh and function space
    auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}},
                                  {32, 16}, mesh::CellType::triangle, part));

    // Would it be possible just to define instead of (T, V, L) (L) alone?
    basix::FiniteElement e = basix::create_element<U>(
        basix::element::family::P,
        mesh::cell_type_to_basix_type(mesh::CellType::triangle), 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    // Create a scalar function space
    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(mesh, e));

    // Create default domain integral on all local cells
    std::int32_t size_local
        = mesh->topology()->index_map(mesh->topology()->dim())->size_local();
    std::vector<std::int32_t> cells(size_local);
    std::iota(cells.begin(), cells.end(), 0);

    // Define element kernel
    auto mass_cell_kernel = [](double*, const double*, const double*,
                               const double*, const int*, const u_int8_t*) {};
    // More automatic type inference?
    const std::map<
        fem::IntegralType,
        std::vector<std::tuple<
            std::int32_t,
            std::function<void(double*, const double*, const double*,
                               const double*, const int*, const u_int8_t*)>,
            std::vector<std::int32_t>>>>
        integrals
        = {{fem::IntegralType::cell, {{-1, mass_cell_kernel, cells}}}};

    // Define form from integral
    // NOTE: Cannot be done through create_form which is recommended in docs.
    auto a = std::make_shared<fem::Form<T>>(
        fem::Form<T>({V, V}, integrals, {}, {}, false, mesh));

    auto sparsity = la::SparsityPattern(
        MPI_COMM_WORLD, {V->dofmap()->index_map, V->dofmap()->index_map},
        {V->dofmap()->index_map_bs(), V->dofmap()->index_map_bs()});
    fem::sparsitybuild::cells(sparsity, cells, {*V->dofmap(), *V->dofmap()});
    sparsity.finalize();
    auto A = la::MatrixCSR<double>(sparsity);

    auto mat_add_values = A.mat_add_values();
    assemble_matrix(mat_add_values, *a, {});
    A.scatter_rev();
  }

  MPI_Finalize();
  return 0;
}
