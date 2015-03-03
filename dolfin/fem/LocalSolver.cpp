// Copyright (C) 2013-2015 Garth N. Wells
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Steven Vandekerckhove, 2014.

#include <array>
#include <memory>
#include <vector>
#include <Eigen/Dense>

#include <dolfin/common/ArrayView.h>
#include <dolfin/common/types.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericLinearAlgebraFactory.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include "assemble.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "UFC.h"
#include "LocalSolver.h"

using namespace dolfin;

//----------------------------------------------------------------------------
LocalSolver::LocalSolver(std::shared_ptr<const Form> a,
                         std::shared_ptr<const Form> L,
                         SolverType solver_type)
  : _a(a), _formL(L), _solver_type(solver_type)
{
  dolfin_assert(a);
  dolfin_assert(a->rank() == 2);
  dolfin_assert(L);
  dolfin_assert(L->rank() == 1);
}
//----------------------------------------------------------------------------
LocalSolver::LocalSolver(std::shared_ptr<const Form> a,
                         SolverType solver_type)
  : _a(a), _solver_type(solver_type)
{
  dolfin_assert(a);
  dolfin_assert(a->rank() == 2);
}
//----------------------------------------------------------------------------
void LocalSolver::solve_global_rhs(Function& u) const
{
  // Compute RHS (global)
  std::shared_ptr<GenericVector> b = u.vector()->factory().create_vector();
  dolfin_assert(b);
  dolfin_assert(_formL);
  assemble(*b, *_formL);

  // Solve local problems
  dolfin_assert(u.vector());
  dolfin_assert(_formL->function_space(0)->dofmap().get());
  solve_local(*u.vector(), *b, *(_formL->function_space(0)->dofmap()));
}
//----------------------------------------------------------------------------
void LocalSolver::solve_local_rhs(Function& u) const
{
  dolfin_assert(_a);
  dolfin_assert(_formL);
  dolfin_assert(_a->rank() == 2);
  dolfin_assert(_formL->rank() == 1);

  // Extract mesh
  dolfin_assert(_a->function_space(0)->mesh());
  const Mesh& mesh = *_a->function_space(0)->mesh();

  // Extract vector
  dolfin_assert(u.vector());
  GenericVector& x = *(u.vector());

 // Create UFC objects
  UFC ufc_a(*_a), ufc_L(*_formL);

  // Check whether to use cache for factorizations
  const bool use_cache = _cholesky_cache.empty()
    and _lu_cache.empty() ? false : true;

  // Get cell integrals
  std::shared_ptr<ufc::cell_integral> integral_a;
  if (!use_cache)
  {
    integral_a = ufc_a.default_cell_integral;
    dolfin_assert(integral_a);
  }
  std::shared_ptr<ufc::cell_integral> integral_L = ufc_L.default_cell_integral;
  dolfin_assert(integral_L);

  // Get dofmaps
  std::array<std::shared_ptr<const GenericDofMap>, 2> dofmaps_a
    = {{_a->function_space(0)->dofmap(), _a->function_space(1)->dofmap()}};
  dolfin_assert(dofmaps_a[0] and dofmaps_a[1]);

  dolfin_assert(_formL->function_space(0)->dofmap());
  std::shared_ptr<const GenericDofMap> dofmap_L
    = _formL->function_space(0)->dofmap();

  // Check dimensions
  dolfin_assert(dofmaps_a[0]->global_dimension()
                == dofmaps_a[1]->global_dimension());
  dolfin_assert(dofmaps_a[0]->global_dimension()
                == dofmap_L->global_dimension());

  // Eigen data structures for local tensors
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_e;
  Eigen::VectorXd b_e, x_e;

  // Eigen factorizations
  Eigen::PartialPivLU<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>> lu;
  Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>> cholesky;

  // Assemble LHS over cells and solve
  Progress p("Performing local (cell-wise) solve", mesh.num_cells());
  ufc::cell ufc_cell;
  std::vector<double> vertex_coordinates;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get local-to-global dof maps for cell
    const ArrayView<const dolfin::la_index> dofs_a0
      = dofmaps_a[0]->cell_dofs(cell->index());
    const ArrayView<const dolfin::la_index> dofs_a1
      = dofmaps_a[1]->cell_dofs(cell->index());
    const ArrayView<const dolfin::la_index> dofs_L
      = dofmap_L->cell_dofs(cell->index());
    dolfin_assert(dofs_a0.size() == dofs_a1.size());
    dolfin_assert(dofs_a0.size() == dofs_L.size());

    // Update to current cell
    cell->get_vertex_coordinates(vertex_coordinates);
    cell->get_cell_data(ufc_cell);

    // Resize local RHS vector and copy global RHS data into local
    // RHS, else compute b_e
    b_e.resize(dofs_L.size());
    ufc_L.update(*cell, vertex_coordinates, ufc_cell,
                 integral_L->enabled_coefficients());

    // Tabulate matrix on cell
    integral_L->tabulate_tensor(b_e.data(), ufc_L.w(),
                                vertex_coordinates.data(),
                                ufc_cell.orientation);

    // Solve local problem
    if (integral_a)
    {
      // Update LHS UFC object
      ufc_a.update(*cell, vertex_coordinates, ufc_cell,
                   integral_a->enabled_coefficients());

      // Resize A_e and tabulate on for cell
      const std::size_t dim = dofmaps_a[0]->num_element_dofs(cell->index());
      dolfin_assert(dim == dofmaps_a[1]->num_element_dofs(cell->index()));
      A_e.resize(dim, dim);
      integral_a->tabulate_tensor(A_e.data(), ufc_a.w(),
                                  vertex_coordinates.data(),
                                  ufc_cell.orientation);
      // Solve local problem
      if (_solver_type == SolverType::Cholesky)
      {
        cholesky.compute(A_e);
        x_e = cholesky.solve(b_e);
      }
      else
      {
        lu.compute(A_e);
        x_e = lu.solve(b_e);
      }
    }
    else
    {
      if (_solver_type == SolverType::Cholesky)
        x_e = _cholesky_cache[cell->index()].solve(b_e);
      else
        x_e = _lu_cache[cell->index()].solve(b_e);
    }

    // Set solution in global vector
    x.set_local(x_e.data(), dofs_a0.size(), dofs_a0.data());

    p++;
  }

  // Finalise vector
  x.apply("insert");
}
//----------------------------------------------------------------------------
void LocalSolver::solve_local(GenericVector& x, const GenericVector& b,
                              const GenericDofMap& dofmap_b) const
{
  dolfin_assert(_a);
  dolfin_assert(_a->rank() == 2);

  // Extract mesh
  dolfin_assert(_a->function_space(0)->mesh());
  const Mesh& mesh = *_a->function_space(0)->mesh();

  // Check whether to use cache for factorizations
  const bool use_cache
    = _cholesky_cache.empty() and _lu_cache.empty() ? false : true;

  // Create UFC object
  UFC ufc_a(*_a);

  // Get cell integral
  std::shared_ptr<ufc::cell_integral> integral_a = ufc_a.default_cell_integral;
  dolfin_assert(integral_a);

  // Get dofmaps
  std::array<std::shared_ptr<const GenericDofMap>, 2> dofmaps_a
    = {{_a->function_space(0)->dofmap(), _a->function_space(1)->dofmap()}};
  dolfin_assert(dofmaps_a[0] and dofmaps_a[1]);

  // Check dimensions
  dolfin_assert(dofmaps_a[0]->global_dimension()
                == dofmaps_a[1]->global_dimension());
  dolfin_assert(dofmaps_a[0]->global_dimension()
                == dofmap_b.global_dimension());

  // Eigen data structures for local tensors
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_e;
  Eigen::VectorXd b_e, x_e;

  // Eigen factorizations
  Eigen::PartialPivLU<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>> lu;
  Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>> cholesky;

  // Assemble LHS over cells and solve
  Progress p("Performing local (cell-wise) solve", mesh.num_cells());
  ufc::cell ufc_cell;
  std::vector<double> vertex_coordinates;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get cell dofmaps
    const ArrayView<const dolfin::la_index> dofs_L
      = dofmap_b.cell_dofs(cell->index());
    const ArrayView<const dolfin::la_index> dofs_a0
      = dofmaps_a[0]->cell_dofs(cell->index());

    // Check dimensions
    dolfin_assert(dofs_L.size() == dofs_a0.size());

    // Copy global RHS data into local RHS
    b_e.resize(dofs_L.size());
    b.get_local(b_e.data(), dofs_L.size(), dofs_L.data());

    // Solve local problem
    if (!use_cache)
    {
      // Update to current cell
      cell->get_vertex_coordinates(vertex_coordinates);
      cell->get_cell_data(ufc_cell);

      // Update LHS UFC object
      ufc_a.update(*cell, vertex_coordinates, ufc_cell,
                   integral_a->enabled_coefficients());

      // Resize A_e and tabulate on for cell
      const std::size_t dim = dofmaps_a[0]->num_element_dofs(cell->index());
      dolfin_assert(dim == dofmaps_a[1]->num_element_dofs(cell->index()));
      A_e.resize(dim, dim);
      integral_a->tabulate_tensor(A_e.data(), ufc_a.w(),
                                  vertex_coordinates.data(),
                                  ufc_cell.orientation);
      // Solve local problem
      if (_solver_type == SolverType::Cholesky)
      {
        cholesky.compute(A_e);
        x_e = cholesky.solve(b_e);
      }
      else
      {
        lu.compute(A_e);
        x_e = lu.solve(b_e);
      }
    }
    else
    {
      if (_solver_type == SolverType::Cholesky)
        x_e = _cholesky_cache[cell->index()].solve(b_e);
      else
        x_e = _lu_cache[cell->index()].solve(b_e);
    }

    // Set solution in global vector
    x.set_local(x_e.data(), dofs_a0.size(), dofs_a0.data());

    p++;
  }

  // Finalise vector
  x.apply("insert");
}
//----------------------------------------------------------------------------
void LocalSolver::factorize()
{
  // Create UFC object
  dolfin_assert(_a);
  UFC ufc(*_a);

  // Check rank
  dolfin_assert(ufc.form.rank() == 2);

  // Raise error for Point integrals
  if (ufc.form.has_vertex_integrals())
  {
    dolfin_error("LocalSolver.cpp",
                 "assemble system",
                 "Point integrals are not supported (yet)");
  }

  // Extract mesh
  const Mesh& mesh = _a->mesh();

  // Collect pointers to dof maps
  std::array<std::shared_ptr<const GenericDofMap>, 2> dofmaps
    = {{_a->function_space(0)->dofmap(), _a->function_space(1)->dofmap()}};
  dolfin_assert(dofmaps[0]);
  dolfin_assert(dofmaps[1]);

  // Get cell integral
  ufc::cell_integral* integral = ufc.default_cell_integral.get();
  dolfin_assert(integral);

  // Resize LU cache
  if (_solver_type==SolverType::Cholesky)
    _cholesky_cache.resize(mesh.num_cells());
  else
    _lu_cache.resize(mesh.num_cells());

  // Eigen data structure for cell matrix
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A;

  // Assemble over cells
  Progress p("Performing local (cell-wise) solve", mesh.num_cells());
  ufc::cell ufc_cell;
  std::vector<double> vertex_coordinates;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update to current cell and update
    cell->get_vertex_coordinates(vertex_coordinates);
    cell->get_cell_data(ufc_cell);
    ufc.update(*cell, vertex_coordinates, ufc_cell,
               integral->enabled_coefficients());

    // Get cell dimension
    std::size_t dim = dofmaps[0]->num_element_dofs(cell->index());

    // Check that local problem is square
    dolfin_assert(dim == dofmaps[1]->num_element_dofs(cell->index()));

    // Resize element matrix
    A.resize(dim, dim);

    // Tabulate A on cell
    integral->tabulate_tensor(A.data(), ufc.w(),
                              vertex_coordinates.data(),
                              ufc_cell.orientation);

     // Compute LU decomposition and store
    if (_solver_type == SolverType::Cholesky)
      _cholesky_cache[cell->index()].compute(A);
    else
      _lu_cache[cell->index()].compute(A);

    p++;
  }
}
//-----------------------------------------------------------------------------
void LocalSolver::clear_factorization()
{
  _cholesky_cache.clear();
  _lu_cache.clear();
}
//-----------------------------------------------------------------------------
