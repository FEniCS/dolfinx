// Custom cell kernel (C++)
//
// This demo shows how to define a custom cell kernel in C++ and have it
// assembled into a dolfinx::la::MatrixCSR.
//
// .. code-block:: cpp

#include <basix/finite-element.h>
#include <basix/mdspan.hpp>
#include <basix/quadrature.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <stdint.h>
#include <utility>
#include <vector>

using namespace dolfinx;

using T = double;

template <typename T, std::size_t ndim>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, ndim>>;
template <typename T>
using mdspan3x3_t
    = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<T,
                                             std::extents<std::size_t, 3, 3>>;
using KernelFn = std::function<void(T*, const T*, const T*, const T*,
                                    const int*, const uint8_t*)>;

// .. code-block:: cpp

/// @brief Compute an element mass matrix on the reference cell.
/// @tparam V Scalar type
/// @param basis Basis functions
/// @param weights Integration weights
/// @return Element reference matrix [data, shape], row-major storage
template <typename V>
std::pair<std::vector<V>, std::array<std::size_t, 2>>
A_ref(mdspan_t<const V, 4> basis, std::span<const V> weights)
{
  std::size_t dim = basis.extent(2);
  std::vector<T> A_b(dim * dim);
  mdspan_t<T, 2> A(A_b.data(), std::array{dim, dim});
  for (std::size_t k = 0; k < basis.extent(1); ++k) // quadrature point
    for (std::size_t i = 0; i < A.extent(0); ++i)   // row i
      for (std::size_t j = 0; j < A.extent(1); ++j) // column j
        A(i, j) += weights[k] * basis(0, k, i, 0) * basis(0, k, j, 0);

  return {A_b, {dim, dim}};
}

/// @brief Compute the RHS vector for f=1 on the reference cell.
/// @tparam V Scalar type
/// @param basis Basis functions
/// @param weights Integration weights
/// @return RHS reference vector
template <typename V>
std::vector<V> b_ref(mdspan_t<const V, 4> basis, std::span<const V> weights)
{
  std::size_t dim = basis.extent(2);
  std::vector<T> b(dim);
  for (std::size_t k = 0; k < basis.extent(1); ++k) // quadrature point
    for (std::size_t i = 0; i < b.size(); ++i)      // row i
      b[i] += weights[k] * basis(0, k, i, 0);
  return b;
}

/// @brief Assemble matrix operator using a std::function kernel
/// function.
/// @tparam T Scalar type
/// @param V Function space,
/// @param kernel Element kernel to execute.
/// @param cells Cells to execute the kernel over.
/// @return Frobenius norm squared of the matrix.
template <typename T>
double assemble_matrix0(std::shared_ptr<fem::FunctionSpace<T>> V, auto kernel,
                        const std::vector<std::int32_t>& cells)
{
  std::map integrals{
      std::pair{fem::IntegralType::cell,
                std::vector{std::tuple{-1, KernelFn(kernel), cells}}}};
  fem::Form<T> a({V, V}, integrals, {}, {}, false, V->mesh());

  auto dofmap = V->dofmap();
  auto sp = la::SparsityPattern(
      MPI_COMM_WORLD, {dofmap->index_map, dofmap->index_map},
      {dofmap->index_map_bs(), dofmap->index_map_bs()});
  fem::sparsitybuild::cells(sp, cells, {*dofmap, *dofmap});
  sp.finalize();
  auto A = la::MatrixCSR<T>(sp);
  common::Timer timer("Assembler0 (matrix)");
  assemble_matrix(A.mat_add_values(), a, {});
  A.scatter_rev();
  return A.squared_norm();
}

/// @brief Assemble a RHS vector using a `std::function` kernel function.
/// @tparam T Scalar type.
/// @param V Function space.
/// @param kernel Element kernel to execute.
/// @param cells Cells to execute the kernel over.
/// @return l2 norm squared of the vector.
template <typename T>
double assemble_vector0(std::shared_ptr<fem::FunctionSpace<T>> V, auto kernel,
                        const std::vector<std::int32_t>& cells)
{
  std::map integrals{
      std::pair{fem::IntegralType::cell,
                std::vector{std::tuple{-1, KernelFn(kernel), cells}}}};
  auto mesh = V->mesh();
  fem::Form<T> L({V}, integrals, {}, {}, false, mesh);
  auto dofmap = V->dofmap();
  la::Vector<T> b(dofmap->index_map, 1);
  common::Timer timer("Assembler0 (vector)");
  fem::assemble_vector(b.mutable_array(), L);
  b.scatter_rev(std::plus<T>());
  return la::squared_norm(b);
}

/// @brief Assemble a matrix operator using an inlined kernel function.
/// @tparam T Scalar type
/// @param V Function space
/// @param kernel Element kernel to execute
/// @param cells Cells to execute the kernel over
/// @return Frobenius norm squared of the matrix.
template <typename T>
double assemble_matrix1(const mesh::Geometry<T>& g, const fem::DofMap& dofmap,
                        auto kernel, std::span<const std::int32_t> cells)
{
  auto sp = la::SparsityPattern(MPI_COMM_WORLD,
                                {dofmap.index_map, dofmap.index_map},
                                {dofmap.index_map_bs(), dofmap.index_map_bs()});
  fem::sparsitybuild::cells(sp, cells, {dofmap, dofmap});
  sp.finalize();
  auto A = la::MatrixCSR<T>(sp);
  auto ident = [](auto, auto, auto, auto) {};
  common::Timer timer("Assembler1 (matrix)");
  fem::impl::assemble_cells(A.mat_add_values(), g.dofmap(), g.x(), cells, ident,
                            dofmap.map(), 1, ident, dofmap.map(), 1, {}, {},
                            kernel, std::span<const T>(), 0, {}, {});
  A.scatter_rev();
  return A.squared_norm();
}

/// @brief Assemble a RHS vector using an inlined kernel function.
/// @tparam T Scalar type.
/// @param V Function space.
/// @param kernel Element kernel to execute.
/// @param cells Cells to execute the kernel over.
/// @return l2 norm squared of the vector.
template <typename T>
double assemble_vector1(const mesh::Geometry<T>& g, const fem::DofMap& dofmap,
                        auto kernel, const std::vector<std::int32_t>& cells)
{
  la::Vector<T> b(dofmap.index_map, 1);
  common::Timer timer("Assembler1 (vector)");
  fem::impl::assemble_cells<T, 1>([](auto, auto, auto, auto) {},
                                  b.mutable_array(), g.dofmap(), g.x(), cells,
                                  dofmap.map(), 1, kernel, {}, {}, 0, {});
  b.scatter_rev(std::plus<T>());
  return la::squared_norm(b);
}

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    // Create mesh
    auto mesh = std::make_shared<mesh::Mesh<T>>(
        mesh::create_rectangle<T>(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}},
                                  {64, 64}, mesh::CellType::triangle));

    // Create Basix element for the field u. This will be used to
    // construct basis functions inside the custom cell kernel.
    constexpr int order = 1;
    basix::FiniteElement e = basix::create_element<T>(
        basix::element::family::P,
        mesh::cell_type_to_basix_type(mesh::CellType::triangle), order,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    // Construct quadrature rule
    constexpr int max_degree = 2 * order;
    auto quadrature_type = basix::quadrature::get_default_rule(
        basix::cell::type::triangle, max_degree);
    auto [X_b, weights] = basix::quadrature::make_quadrature<T>(
        quadrature_type, basix::cell::type::triangle,
        basix::polyset::type::standard, max_degree);
    mdspan_t<const T, 2> X(X_b.data(), weights.size(), 2);

    // Create a scalar function space
    auto V = std::make_shared<fem::FunctionSpace<T>>(
        fem::create_functionspace(mesh, e));

    // Build list of cells to assembler over (all cells owned by this
    // rank)
    std::int32_t size_local
        = mesh->topology()->index_map(mesh->topology()->dim())->size_local();
    std::vector<std::int32_t> cells(size_local);
    std::iota(cells.begin(), cells.end(), 0);

    // Tabulate basis functions at quadrature points
    auto tabulate_shape = e.tabulate_shape(0, weights.size());
    std::size_t length = std::accumulate(
        tabulate_shape.begin(), tabulate_shape.end(), 1, std::multiplies<>{});
    std::vector<T> basis_b(length);
    mdspan_t<T, 4> basis(basis_b.data(), tabulate_shape);
    e.tabulate(0, X, basis);

    // Utility function to compute det(J) for an affine triangle cell
    auto detJ = [](const T* x)
    {
      return std::abs((x[0] - x[3]) * (x[7] - x[4])
                      - (x[1] - x[4]) * (x[6] - x[3]));
    };

    // Finite element mass matrix kernel function
    auto [A_hat_b, A_shape] = A_ref<T>(basis, weights);
    auto kernel_a
        = [A_hat = mdspan3x3_t<T>(A_hat_b.data()), detJ](
              T* A, const T*, const T*, const T* x, const int*, const uint8_t*)
    {
      T scale = detJ(x);
      mdspan3x3_t<T> _A(A);
      for (std::size_t i = 0; i < A_hat.extent(0); ++i)
        for (std::size_t j = 0; j < A_hat.extent(1); ++j)
          _A(i, j) = scale * A_hat(i, j);
    };

    // Finite element RHD (f=1_ kernel function
    auto kernel_L
        = [b_hat = b_ref<T>(basis, weights), detJ](
              T* b, const T*, const T*, const T* x, const int*, const uint8_t*)
    {
      T scale = detJ(x);
      for (std::size_t i = 0; i < 3; ++i)
        b[i] = scale * b_hat[i];
    };

    // Assemble matrix and vector using std::function kernel
    assemble_matrix0<T>(V, kernel_a, cells);
    assemble_vector0<T>(V, kernel_L, cells);

    // Assemble matrix and vector using lambda kernel. This version
    // supports efficient inlining of the kernel.
    assemble_matrix1<T>(V->mesh()->geometry(), *V->dofmap(), kernel_a, cells);
    assemble_vector1<T>(V->mesh()->geometry(), *V->dofmap(), kernel_L, cells);
  }

  list_timings(MPI_COMM_WORLD, {TimingType::wall});

  MPI_Finalize();
  return 0;
}
