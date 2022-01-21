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
        MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {8, 8},
        mesh::CellType::triangle, mesh::GhostMode::none));

    auto V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_poisson_M, "ui", mesh));

    // Prepare and set Constants for the bilinear form
    auto f = std::make_shared<fem::Constant<T>>(6.0);

    // Define variational forms
    auto L = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_poisson_L, {V}, {}, {{"f", f}}, {}));

    // Action of the bilinear form "a" on a function ui
    auto ui = std::make_shared<fem::Function<T>>(V);
    auto M = std::make_shared<fem::Form<T>>(
        fem::create_form<T>(*form_poisson_M, {V}, {{"ui", ui}}, {{}}, {}));

    auto u_D = std::make_shared<fem::Function<T>>(V);
    auto facets = mesh::exterior_facet_indices(*mesh);
    const auto bdofs = fem::locate_dofs_topological({*V}, 1, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(u_D, bdofs);

    // Assemble RHS vector
    la::Vector<T> b(V->dofmap()->index_map, V->dofmap()->index_map_bs());
    fem::assemble_vector(b.mutable_array(), *L);

    // Apply lifting to account for Dirichlet boundary condition
    fem::set_bc(ui->x()->mutable_array(), {bc}, -1.0);
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
    auto u = std::make_shared<fem::Function<T>>(V);
    linalg::cg(*u->x(), b, action, 200, 1e-6);

    // Set BC values in the solution vectors
    fem::set_bc(u->x()->mutable_array(), {bc}, 1.0);

    auto E = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *form_poisson_E, {}, {{"ue", u_D}, {"uc", u}}, {}, {}, mesh));
    T error = fem::assemble_scalar(*E);
    std::cout << error;
  }

  common::subsystem::finalize_mpi();
  return 0;
}
