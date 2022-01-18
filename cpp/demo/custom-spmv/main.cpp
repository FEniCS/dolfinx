#include "poisson.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/la/MatrixCSR.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;
using T = PetscScalar;

namespace
{

template <typename T>
void spmv_impl(xtl::span<const T> values,
               xtl::span<const std::int32_t> row_begin,
               xtl::span<const std::int32_t> row_end,
               xtl::span<const std::int32_t> indices, xtl::span<const T> x,
               xtl::span<T> y)
{
  assert(row_begin.size() == row_end.size());
  for (std::size_t i = 0; i < row_begin.size(); i++)
  {
    double vi{0};
    for (std::int32_t j = row_begin[i]; j < row_end[i]; j++)
    {
      vi += values[j] * x[indices[j]];
    }
    y[i] += vi;
  }
}
} // namespace

int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);

  {
    MPI_Comm comm = MPI_COMM_WORLD;

    // Create mesh and function space
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_box(
        comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {10, 10, 10},
        mesh::CellType::tetrahedron, mesh::GhostMode::none));

    auto V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_poisson_a, "u", mesh));

    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<fem::Constant<T>>(2.0);
    auto f = std::make_shared<fem::Function<T>>(V);
    auto ui = std::make_shared<fem::Function<T>>(V);

    // Define variational forms
    auto a = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_a, {V, V}, {}, {{"kappa", kappa}}, {}));
    auto L = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_poisson_L, {V}, {{"f", f}}, {}, {}));
    auto M = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_M, {V}, {{"ui", ui}}, {{"kappa", kappa}}, {}));

    f->interpolate([](auto& x) -> xt::xarray<T> { return xt::row(x, 0); });

    // Create sparsity pattern
    la::SparsityPattern sp = fem::create_sparsity_pattern(*a);
    sp.assemble();

    // Assemble matrix
    la::MatrixCSR<double> A(sp);
    fem::assemble_matrix(la::MatrixCSR<T>::mat_add_values(A), *a, {});
    A.finalize();

    // Create vectors compatible with the sparse matrix
    auto column_map = A.index_maps()[1];
    la::Vector<T> x(column_map, 1);
    la::Vector<T> y0(column_map, 1);
    la::Vector<T> y1(column_map, 1);

    // Compute input vectors
    fem::assemble_vector(x.mutable_array(), *L);
    std::copy_n(x.array().begin(), ui->x()->array().size(),
                ui->x()->mutable_array().begin());

    // The matrix A is distributed across P  processes by blocks of rows:
    //  A = |   A_0  |
    //      |   A_1  |
    //      |   ...  |
    //      |  A_P-1 |
    //
    // Each submatrix A_i is owned by a single process "i" and can be further
    // decomposed into diagonal and off diagonal blocks:
    //  Ai = |Ai_diag Ai_off|
    //
    // If A is square, the diagonal block Ai_diag is also square and countains
    // only owned columns and rows.
    //
    // The block Ai_off contains ghost columns (unowned dofs).
    //

    // Create function to compute y = A x in parallel
    auto spmv = [&A](la::Vector<T>& x, la::Vector<T>& y)
    {
      // start communication (update ghosts)
      x.scatter_fwd_begin();

      xtl::span<const std::int32_t> row_ptr = A.row_ptr();
      xtl::span<const std::int32_t> cols = A.cols();
      xtl::span<const std::int32_t> off_diag_offset = A.off_diag_offset();
      xtl::span<const T> values = A.values();

      std::int32_t nrows = A.rows();

      xtl::span<const T> _x = x.array();
      xtl::span<T> _y = y.mutable_array();

      xtl::span<const std::int32_t> row_begin(row_ptr.data(), nrows);
      xtl::span<const std::int32_t> row_end(row_ptr.data() + 1, nrows);

      // First stage:  spmv - diagonal
      // yi += Ai_diag * xi
      spmv_impl<T>(values, row_begin, off_diag_offset, cols, _x, _y);

      // finalize ghost update
      x.scatter_fwd_end();

      // Second stage:  spmv - off-diagonal
      // yi += Ai_off * xi
      spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
    };

    // Two stage matrix vector computation
    common::Timer t0("~SPMV");
    spmv(x, y0);
    t0.stop();

    ui->x()->scatter_fwd();
    const auto coefficients = fem::pack_coefficients(*M);
    const std::vector<T> constants = fem::pack_constants(*M);
    common::Timer t1("~Matrix Free action");
    {
      fem::assemble_vector(y1.mutable_array(), *M, tcb::make_span(constants),
                           make_coefficients_span(coefficients));
      y1.scatter_rev(common::IndexMap::Mode::add);
    }
    t1.stop();

    std::int32_t ndofs_local = column_map->size_local();

    // Check solutions
    double err_abs = 0;
    const auto y0_array = y0.array();
    const auto y1_array = y1.array();
    for (int i = 0; i < ndofs_local; i++)
      err_abs += std::abs(y0_array[i] - y1_array[i]);

    if (err_abs > 1e-15)
      throw std::runtime_error("Solution mismatch");

    dolfinx::list_timings(comm, {dolfinx::TimingType::wall});
  }

  common::subsystem::finalize_mpi();
  return 0;
}
