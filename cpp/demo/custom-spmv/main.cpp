#include "poisson.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/la/MatrixCSR.h>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;
using T = PetscScalar;

// Then follows the definition of the coefficient functions (for
// :math:`f` and :math:`g`), which are derived from the
// :cpp:class:`Expression` class in DOLFINx
//
// .. code-block:: cpp

// Inside the ``main`` function, we begin by defining a mesh of the
// domain. As the unit square is a very standard domain, we can use a
// built-in mesh provided by the :cpp:class:`UnitSquareMesh` factory. In
// order to create a mesh consisting of 32 x 32 squares with each square
// divided into two triangles, and the finite element space (specified in
// the form file) defined relative to this mesh, we do as follows
//
// .. code-block:: cpp

namespace
{

/// Compute y = Ai x
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
    auto mesh = std::make_shared<mesh::Mesh>(
        mesh::create_box(comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {2, 2, 2},
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

    la::MatrixCSR<double> A(sp);
    fem::assemble_matrix(la::MatrixCSR<T>::mat_add_values(A), *a, {});
    A.finalize();

    // Create compatible vectors
    auto column_map = A.index_maps()[1];
    la::Vector<T> x(column_map, 1);
    fem::assemble_vector(x.mutable_array(), *L);

    std::copy_n(x.array().begin(), ui->x()->array().size(),
                ui->x()->mutable_array().begin());

    la::Vector<T> y(column_map, 1);

    // Compute diagonal
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
      spmv_impl<T>(values, row_begin, off_diag_offset, cols, _x, _y);

      // finalize ghost update
      x.scatter_fwd_end();

      // Second stage:  spmv - off-diagonal
      // spmv_impl<T>(values, off_diag_offset, row_end, cols, _x, _y);
    };

    // Two stage matrix vector computation
    common::Timer t0("~SPMV");
    spmv(x, y);
    t0.stop();

    la::Vector<T> y1(column_map, 1);

    common::Timer t1("~Matrix Free action");
    ui->x()->scatter_fwd();
    const auto coefficients = fem::pack_coefficients(*M);
    const std::vector<T> constants = fem::pack_constants(*M);
    fem::assemble_vector(y1.mutable_array(), *M, tcb::make_span(constants),
                         make_coefficients_span(coefficients));
    y1.scatter_rev(common::IndexMap::Mode::add);
    t1.stop();

    std::cout << y.norm() << " " << y1.norm() << std::endl;

    // dolfinx::list_timings(comm, {dolfinx::TimingType::wall});
  }

  common::subsystem::finalize_mpi();
  return 0;
}
