// Custom cell kernel (C++)
// .. code-block:: cpp

#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx.h>
#include <utility>
#include <vector>

using namespace dolfinx;

using T = double;
using U = typename dolfinx::scalar_value_type_t<T>;

// .. code-block:: cpp

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);

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
    std::int32_t size_local = mesh->topology()->index_map(2)->size_local();
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
    auto a = fem::Form<T>({V, V}, integrals, {}, {}, false, mesh);
  }

  return 0;
}