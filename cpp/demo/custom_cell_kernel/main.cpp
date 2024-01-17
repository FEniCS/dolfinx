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
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, ndim>>;

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
    const int num_points = weights.size();
    mdspan_t<const T, 2> points_span(points.data(), num_points, 2);

    // Create a scalar function space
    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(mesh, e));

    // Create default domain integral on all local cells
    std::int32_t size_local
        = mesh->topology()->index_map(mesh->topology()->dim())->size_local();
    std::vector<std::int32_t> cells(size_local);
    std::iota(cells.begin(), cells.end(), 0);

    // Basis element tabulation exploration
    const auto tabulate_shape = e.tabulate_shape(0, num_points);
    const auto length = std::accumulate(
        tabulate_shape.begin(), tabulate_shape.end(), 1, std::multiplies<>{});
    std::vector<T> basis(length);
    mdspan_t<T, 4> basis_span(basis.data(), tabulate_shape);

    // Tabulate basis functions
    e.tabulate(0, points_span, basis_span);

    // Calculate mass matrix on reference cell
    const auto e_dim = e.dim();
    std::vector<T> A_hat(e_dim * e_dim);
    mdspan_t<T, 2> A_hat_span(A_hat.data(), e.dim(), e.dim());

    // einsum k,ki,kj->ij on weights, basis_span, basis_span
    for (std::size_t k = 0; k < weights.size(); ++k)
    {
      for (std::size_t i = 0; i < A_hat_span.extent(0); ++i)
      {
        for (std::size_t j = 0; j < A_hat_span.extent(1); ++j)
        {
          A_hat_span(i, j)
              += weights[k] * basis_span(0, k, i, 0) * basis_span(0, k, j, 0);
        }
      }
    }

    // Define element kernel
    std::function<void(T*, const T*, const T*, const U*, const int*,
                       const u_int8_t*)>
        mass_cell_kernel = [&A_hat_span](T* A_cell, const T*, const T*,
                                         const U*, const int*, const u_int8_t*)
    {
      // TODO: Calculate detJ and scale
      for (std::size_t i = 0; i < A_hat_span.extent(0); ++i)
      {
        for (std::size_t j = 0; j < A_hat_span.extent(1); ++j)
        {
          A_cell[i * A_hat_span.extent(0) + j] = A_hat_span(i, j);
        }
      }
    };
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
    auto A = la::MatrixCSR<T>(sparsity);

    auto mat_add_values = A.mat_add_values();
    assemble_matrix(mat_add_values, *a, {});
    A.scatter_rev();
  }

  MPI_Finalize();
  return 0;
}
