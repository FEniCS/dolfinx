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
#include <utility>
#include <vector>

using namespace dolfinx;

using T = double;                                   // field scalar type
using U = typename dolfinx::scalar_value_type_t<T>; // geometry scalar type

template <typename T, std::size_t ndim>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, ndim>>;

// .. code-block:: cpp

template <typename V>
std::pair<std::vector<V>, std::array<std::size_t, 2>>
A_ref(mdspan_t<const V, 4> basis, std::span<const V> weights)
{
  std::size_t dim = basis.extent(2);
  std::array shape = {dim, dim};
  std::vector<T> A_b(shape[0] * shape[1]);
  mdspan_t<T, 2> A(A_b.data(), shape);

  // einsum k,ki,kj->ij on weights, basis_span, basis_span
  for (std::size_t k = 0; k < basis.extent(1); ++k)
    for (std::size_t i = 0; i < A.extent(0); ++i)
      for (std::size_t j = 0; j < A.extent(1); ++j)
        A(i, j) += weights[k] * basis(0, k, i, 0) * basis(0, k, j, 0);

  return {A_b, shape};
}

template <typename V>
std::vector<V> b_ref(mdspan_t<const V, 4> basis, std::span<const V> weights)
{
  std::size_t dim = basis.extent(2);
  std::vector<T> b(dim);
  for (std::size_t k = 0; k < basis.extent(1); ++k)
    for (std::size_t i = 0; i < b.size(); ++i)
      b[i] += weights[k] * basis(0, k, i, 0);
  return b;
}

template <typename T>
void assemble0(std::shared_ptr<fem::FunctionSpace<T>> V, auto kernel,
               const std::vector<std::int32_t>& cells)
{
  // Define form from integral
  using KernelFn = std::function<void(T*, const T*, const T*, const T*,
                                      const int*, const u_int8_t*)>;
  std::map integrals{
      std::pair{fem::IntegralType::cell,
                std::vector{std::tuple{-1, KernelFn(kernel), cells}}}};

  std::shared_ptr<const mesh::Mesh<U>> mesh = V->mesh();
  fem::Form<T> a({V, V}, integrals, {}, {}, false, mesh);

  auto dofmap = V->dofmap();
  auto sparsity = la::SparsityPattern(
      MPI_COMM_WORLD, {dofmap->index_map, dofmap->index_map},
      {dofmap->index_map_bs(), dofmap->index_map_bs()});
  fem::sparsitybuild::cells(sparsity, cells, {*dofmap, *dofmap});
  sparsity.finalize();
  auto A = la::MatrixCSR<T>(sparsity);
  {
    common::Timer timer("Assembler0");
    for (int i = 0; i < 100; ++i)
      assemble_matrix(A.mat_add_values(), a, {});
    A.scatter_rev();
  }
  std::cout << "Norm (0): " << A.squared_norm() << std::endl;
}

template <typename T>
void assemble_vector0(std::shared_ptr<fem::FunctionSpace<T>> V, auto kernel,
                      const std::vector<std::int32_t>& cells)
{
  // Define form from integral
  using KernelFn = std::function<void(T*, const T*, const T*, const T*,
                                      const int*, const u_int8_t*)>;
  std::map integrals{
      std::pair{fem::IntegralType::cell,
                std::vector{std::tuple{-1, KernelFn(kernel), cells}}}};

  std::shared_ptr<const mesh::Mesh<U>> mesh = V->mesh();
  fem::Form<T> L({V}, integrals, {}, {}, false, mesh);

  auto dofmap = V->dofmap();
  la::Vector<T> b(dofmap->index_map, 1);
  {
    common::Timer timer("Assembler_vector0");
    for (int i = 0; i < 100; ++i)
      fem::assemble_vector<T>(b.mutable_array(), L);
    b.scatter_rev(std::plus<T>());
  }
  // std::cout << "Norm vector (0): " << b.norm() << std::endl;
}

template <typename T>
void assemble1(const fem::FunctionSpace<T>& V, auto kernel,
               std::span<const std::int32_t> cells)
{
  auto mesh = V.mesh();
  auto dofmap = V.dofmap();
  auto sparsity = la::SparsityPattern(
      MPI_COMM_WORLD, {dofmap->index_map, dofmap->index_map},
      {dofmap->index_map_bs(), dofmap->index_map_bs()});
  fem::sparsitybuild::cells(sparsity, cells, {*dofmap, *dofmap});
  sparsity.finalize();
  auto A = la::MatrixCSR<T>(sparsity);

  auto ident = [](auto, auto, auto, auto) {};
  const mesh::Geometry<U>& g = mesh->geometry();
  common::Timer timer("Assembler1");
  for (int i = 0; i < 100; ++i)
    fem::impl::assemble_cells(A.mat_add_values(), g.dofmap(), g.x(), cells,
                              ident, dofmap->map(), 1, ident, dofmap->map(), 1,
                              {}, {}, kernel, std::span<const T>(), 0, {}, {});
  A.scatter_rev();
  std::cout << "Norm (1): " << A.squared_norm() << std::endl;
}

template <typename T>
void assemble_vector1(std::shared_ptr<fem::FunctionSpace<T>> V, auto kernel,
                      const std::vector<std::int32_t>& cells)
{
  auto mesh = V.mesh();
  auto dofmap = V.dofmap();
  la::Vector<T> b(dofmap->index_map, 1);

  auto ident = [](auto, auto, auto, auto) {};
  const mesh::Geometry<U>& g = mesh->geometry();
  common::Timer timer("Assembler1");
  for (int i = 0; i < 100; ++i)
    fem::impl::assemble_cells(ident, b.mutable_array(), g.dofmap(), g.x(),
                              cells, dofmap->map(), 1, kernel,
                              std::span<const T>(), std::span<const T>(), 0,
                              std::span<const std::uint32_t>());

  // b.scatter_rev();
  // std::cout << "Norm (1): " << b.squared_norm() << std::endl;
}

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    // Create mesh and function space
    auto mesh = std::make_shared<mesh::Mesh<U>>(
        mesh::create_rectangle<U>(MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}},
                                  {64, 64}, mesh::CellType::triangle));

    // Create Basix element for the field u. This will be used to construct
    // basis functions inside the custom cell kernel.
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
    const auto num_points = static_cast<std::size_t>(weights.size());
    mdspan_t<const T, 2> X(X_b.data(), num_points, 2);

    // Create a scalar function space
    auto V = std::make_shared<fem::FunctionSpace<U>>(
        fem::create_functionspace(mesh, e));

    // Create default domain integral on all local cells
    std::int32_t size_local
        = mesh->topology()->index_map(mesh->topology()->dim())->size_local();
    std::vector<std::int32_t> cells(size_local);
    std::iota(cells.begin(), cells.end(), 0);

    // Tabulate basis functions
    auto tabulate_shape = e.tabulate_shape(0, num_points);
    const std::size_t length = std::accumulate(
        tabulate_shape.begin(), tabulate_shape.end(), 1, std::multiplies<>{});
    std::vector<T> basis_b(length);
    mdspan_t<T, 4> basis(basis_b.data(), tabulate_shape);
    e.tabulate(0, X, basis);

    // Define finite element mass kernel.
    auto [A_hat_b, A_shape] = A_ref<U>(basis, weights);
    mdspan_t<T, 2> A_hat(A_hat_b.data(), A_shape);
    auto mass_cell_kernel = [A_hat](T* A, const T*, const T*, const U* x,
                                    const int*, const u_int8_t*)
    {
      U detJ = std::abs((x[0] - x[3]) * (x[7] - x[4])
                        - (x[1] - x[4]) * (x[6] - x[3]));
      for (std::size_t i = 0; i < A_hat.extent(0); ++i)
        for (std::size_t j = 0; j < A_hat.extent(1); ++j)
          A[i * A_hat.extent(0) + j] = detJ * A_hat(i, j);
    };

    auto b_hat = b_ref<U>(basis, weights);
    auto mass_cell_kernel_rhs = [b_hat](T* A, const T*, const T*, const U* x,
                                        const int*, const u_int8_t*)
    {
      U detJ = std::abs((x[0] - x[3]) * (x[7] - x[4])
                        - (x[1] - x[4]) * (x[6] - x[3]));
      for (std::size_t i = 0; i < b_hat.size(); ++i)
        A[i] = detJ * b_hat[i];
    };

    assemble0<T>(V, mass_cell_kernel, cells);
    assemble_vector0<T>(V, mass_cell_kernel, cells);

    assemble1<T>(*V, mass_cell_kernel_rhs, cells);
  }

  list_timings(MPI_COMM_WORLD, {TimingType::wall});

  MPI_Finalize();
  return 0;
}
