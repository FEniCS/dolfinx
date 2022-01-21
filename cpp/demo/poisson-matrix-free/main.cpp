// Copyright (C) 2022 Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "poisson.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <xtensor/xarray.hpp>

using namespace dolfinx;

using T = PetscScalar;

namespace linalg
{
/// Compute vector r = alpha*x + y
/// @param r Result
/// @param alpha
/// @param x
/// @param y
template <typename T>
void axpy(la::Vector<T>& r, T alpha, const la::Vector<T>& x,
          const la::Vector<T>& y)
{
  std::transform(x.array().cbegin(),
                 std::next(x.array().cbegin(), x.map()->size_local()),
                 y.array().cbegin(), r.mutable_array().begin(),
                 [alpha](auto x, auto y) { return alpha * x + y; });
}

/// Solve problem A.x = b using the Conjugate Gradient method
/// @param b RHS Vector
/// @param x Solution Vector
/// @param matvec_function Function that provides the operator action
/// @param kmax Maxmimum number of iterations
template <typename T>
int cg(la::Vector<T>& x, const la::Vector<T>& b,
       std::function<void(la::Vector<T>&, la::Vector<T>&)> matvec_function,
       int kmax = 50, double rtol = 1e-8)
{
  int M = b.map()->size_local();
  MPI_Comm comm = b.map()->comm(common::IndexMap::Direction::forward);
  int rank = dolfinx::MPI::rank(comm);

  // Working vectors Residual vector
  la::Vector<T> r(b), y(b), p(x);
  std::copy_n(r.array().begin(), M, p.mutable_array().begin());

  double rnorm0 = r.squared_norm();

  // Iterations of CG
  const double rtol2 = rtol * rtol;
  double rnorm = rnorm0;
  int k = 0;
  while (k < kmax)
  {
    ++k;

    // MatVec (y = A p)
    matvec_function(p, y);

    // alpha = r.r/p.y
    const T alpha = rnorm / la::inner_product(p, y);

    // Update x (x <- x + alpha*p)
    axpy(x, alpha, p, x);

    // Update r (r <- r - alpha*y)
    axpy(r, -alpha, y, r);

    // Update residual norm
    // Note: we use T for beta to support float, double, etc. T can be
    // complex, despite its value always being real
    const double rnorm_new = r.squared_norm();
    const T beta = rnorm_new / rnorm;
    rnorm = rnorm_new;

    if (rank == 0)
      std::cout << "Iteration: " << k << ": " << std::sqrt(rnorm / rnorm0)
                << std::endl;

    if (rnorm / rnorm0 < rtol2)
      break;

    // Update p (p <- beta*p + r)
    axpy(p, beta, p, r);
  }

  return k;
}
} // namespace linalg

int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);

  {
    // Create mesh and function space
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_rectangle(
        MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}}, {32, 16},
        mesh::CellType::triangle, mesh::GhostMode::none));

    auto V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_poisson_a, "u", mesh));

    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<fem::Constant<T>>(2.0);
    auto f = std::make_shared<fem::Function<T>>(V);
    auto g = std::make_shared<fem::Function<T>>(V);

    // Define variational forms
    auto L = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_L, {V}, {{"f", f}, {"g", g}}, {}, {}));

    // Action of the bilinear form "a" on a function ui
    auto ui = std::make_shared<fem::Function<T>>(V);
    auto M = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_M, {V}, {{"ui", ui}}, {{"kappa", kappa}}, {}));

    auto facets = mesh::locate_entities_boundary(
        *mesh, 1,
        [](auto& x) -> xt::xtensor<bool, 1>
        {
          auto x0 = xt::row(x, 0);
          return xt::isclose(x0, 0.0) or xt::isclose(x0, 2.0);
        });
    const auto bdofs = fem::locate_dofs_topological({*V}, 1, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(1.0, bdofs, V);

    f->interpolate(
        [](auto& x) -> xt::xarray<T>
        {
          auto dx = xt::square(xt::row(x, 0) - 0.5)
                    + xt::square(xt::row(x, 1) - 0.5);
          return 10 * xt::exp(-(dx) / 0.02);
        });

    g->interpolate(
        [](auto& x) -> xt::xarray<T> { return xt::sin(5 * xt::row(x, 0)); });

    // Compute solution
    fem::Function<T> u(V);
    la::Vector<T> b(V->dofmap()->index_map, V->dofmap()->index_map_bs());

    // Assemble RHS vector
    fem::assemble_vector(b.mutable_array(), *L);

    // Apply lifting to account for Dirichlet boundary condition
    fem::set_bc(u.x()->mutable_array(), {bc}, 0.0);
    dolfinx::fem::assemble_vector(b.mutable_array(), *M);

    // Communicate ghost values
    b.scatter_rev(common::IndexMap::Mode::add);

    // Set BC dofs to zero (effectively zeroes columns of A)
    fem::set_bc(b.mutable_array(), {bc}, 0.0);

    // Create function for computing the action of A on x (y = Ax)
    std::function<void(la::Vector<T>&, la::Vector<T>&)> action
        = [&](la::Vector<T>& x, la::Vector<T>& y)
    {
      // Update ghost values and zero y
      x.scatter_fwd();
      y.set(0);

      // Update coefficient ui (just copy data from x to ui)
      std::copy(x.array().begin(), x.array().end(),
                ui->x()->mutable_array().begin());

      // Compute action of A on x
      dolfinx::fem::assemble_vector(y.mutable_array(), *M);

      // Set BC dofs to zero (effectively zeroes rows of A)
      fem::set_bc(y.mutable_array(), {bc}, 0.0);

      // Communicate ghost values
      y.scatter_rev(common::IndexMap::Mode::add);
    };

    // Compute solution using the conjugate gradient method
    u.x()->set(0);
    linalg::cg(*u.x(), b, action, 100, 1e-6);

    // Set BC values in the solution vectors
    fem::set_bc(u.x()->mutable_array(), {bc}, 1.0);

    // Save solution in VTK format
    // io::VTKFile file(mesh->comm(), "u.pvd", "w");
    // file.write({u}, 0.0);
  }

  common::subsystem::finalize_mpi();
  return 0;
}
