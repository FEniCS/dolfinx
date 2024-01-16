// Custom cell kernel (C++)
// .. code-block:: cpp

#include <iostream>

#include <basix/finite-element.h>
#include <basix/mdspan.hpp>
#include <basix/quadrature.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <utility>
#include <vector>

using namespace dolfinx;

using T = double; // field scalar type
using U = double; // geometry scalar type

template <typename T, std::size_t ndim>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, ndim>>;

// .. code-block:: cpp

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    // Create mesh and function space
    const auto part
        = mesh::create_cell_partitioner(mesh::GhostMode::shared_facet);
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}},
                                  {32, 16}, mesh::CellType::triangle, part));

    // Create basix element for the field u. This will be used to construct
    // basis functions inside the custom cell kernel.
    const basix::FiniteElement e = basix::create_element<T>(
        basix::element::family::P,
        mesh::cell_type_to_basix_type(mesh::CellType::triangle), 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    const int max_degree = 2;
    const auto quadrature_type = basix::quadrature::get_default_rule(
        basix::cell::type::triangle, max_degree);
    const auto [points, weights] = basix::quadrature::make_quadrature<T>(
        quadrature_type, basix::cell::type::triangle,
        basix::polyset::type::standard, max_degree);
    mdspan_t<T, 2> points_span(points.data(), weights.size(), 2);

    for (T i: points)
      std::cout << i << ' ';
    std::cout << std::endl;
    for (T i: weights)
      std::cout << i << ' ';

    // Create a scalar function space
    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(mesh, e));

    // Create default domain integral on all local cells
    std::int32_t size_local
        = mesh->topology()->index_map(mesh->topology()->dim())->size_local();
    std::vector<std::int32_t> cells(size_local);
    std::iota(cells.begin(), cells.end(), 0);

    // Basis element tabulation exploration
    const auto tabulate_shape = e.tabulate_shape(0, 3);
    const auto length
        = std::accumulate(std::begin(tabulate_shape), std::end(tabulate_shape),
                          0, std::multiplies<>{});
    std::vector<T> basis(length);
    mdspan_t<T, 4> basis_span(basis.data(), tabulate_shape);

    // Tabulate basis functions
    e.tabulate(0, points_span, basis_span);  

    // Define element kernel
    std::function<void(T*, const T*, const T*, const U*, const int*,
                       const u_int8_t*)>
        mass_cell_kernel
        = [](T*, const T*, const T*, const U*, const int*, const u_int8_t*) {};
    const std::map integrals{
        std::pair{fem::IntegralType::cell,
                  std::vector{std::tuple{-1, mass_cell_kernel, cells}}}};

    // Define form from integral
    const auto a = std::make_shared<fem::Form<T>>(
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
