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
// Modified by Steven Vandekerckhove, 2014
// Modified by Tormod Landet, 2015

#include <array>
#include <memory>
#include <vector>
#include <Eigen/Dense>

#include <dolfin/common/ArrayView.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/types.h>
#include <dolfin/fem/LocalAssembler.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericLinearAlgebraFactory.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/log.h>
#include <dolfin/log/Progress.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include "assemble.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "UFC.h"
#include "LocalSolver.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------
LocalSolver::LocalSolver(std::shared_ptr<const Form> a, SolverType solver_type)
  : _a(a), _solver_type(solver_type)
{
  dolfin_assert(a);
  dolfin_assert(a->rank() == 2);
}
//-----------------------------------------------------------------------------
void LocalSolver::solve_global_rhs(Function& u) const
{
  // Compute RHS (global)
  std::shared_ptr<GenericVector> b
    = u.vector()->factory().create_vector(u.vector()->mpi_comm());
  dolfin_assert(b);
  dolfin_assert(_formL);
  assemble(*b, *_formL);

  // Extract the vector where the solution will be stored
  dolfin_assert(u.vector());
  GenericVector& x = *(u.vector());

  // Solve local problems
  _solve_local(x, b.get(), nullptr);
}
//-----------------------------------------------------------------------------
void LocalSolver::solve_local_rhs(Function& u) const
{
  // Extract the vector where the solution will be stored
  dolfin_assert(u.vector());
  GenericVector& x = *(u.vector());

  // Loop over all cells and assemble local LHS & RHS which are then solved
  _solve_local(x, nullptr, nullptr);
}
//-----------------------------------------------------------------------------
void LocalSolver::solve_local(GenericVector& x, const GenericVector& b,
                              const GenericDofMap& dofmap_b) const
{
  _solve_local(x, &b, &dofmap_b);
}
//-----------------------------------------------------------------------------
void LocalSolver::_solve_local(GenericVector& x, const GenericVector* global_b,
                               const GenericDofMap* dofmap_L) const
{
  // Check that we have valid bilinear form
  dolfin_assert(_a);
  dolfin_assert(_a->rank() == 2);

  // Set timer
  Timer timer("Solve local problems");

  // Create UFC objects
  UFC ufc_a(*_a);
  std::unique_ptr<UFC> ufc_L;

  // Check that we have valid linear form or a dofmap for it
  if (dofmap_L)
    dolfin_assert(global_b);
  else
  {
    dolfin_assert(_formL);
    dolfin_assert(_formL->rank() == 1);
    dolfin_assert(_formL->function_space(0)->dofmap());
    dofmap_L = _formL->function_space(0)->dofmap().get();
    ufc_L.reset(new UFC(*_formL));
  }

  // Extract the mesh
  dolfin_assert(_a->function_space(0)->mesh());
  const Mesh& mesh = *_a->function_space(0)->mesh();

  // Get bilinear form dofmaps
  std::array<std::shared_ptr<const GenericDofMap>, 2> dofmaps_a
    = {{_a->function_space(0)->dofmap(), _a->function_space(1)->dofmap()}};
  dolfin_assert(dofmaps_a[0] and dofmaps_a[1]);

  // Extract cell_domains etc from left-hand side form
  const MeshFunction<std::size_t>* cell_domains
    = _a->cell_domains().get();
  const MeshFunction<std::size_t>* exterior_facet_domains
    = _a->exterior_facet_domains().get();
  const MeshFunction<std::size_t>* interior_facet_domains
    = _a->interior_facet_domains().get();

  // Eigen data structures and factorisations for cell data structures
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                Eigen::RowMajor> A_e, b_e;
  Eigen::VectorXd x_e;
  Eigen::PartialPivLU<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>> lu;
  Eigen::LLT<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>> cholesky;
  bool use_cache = !(_cholesky_cache.empty() and _lu_cache.empty());

  // Loop over cells and solve local problems
  Progress p("Performing local (cell-wise) solve", mesh.num_cells());
  ufc::cell ufc_cell;
  std::vector<double> coordinate_dofs;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get local-to-global dof maps for cell
    const ArrayView<const dolfin::la_index> dofs_a0
      = dofmaps_a[0]->cell_dofs(cell->index());
    const ArrayView<const dolfin::la_index> dofs_a1
      = dofmaps_a[1]->cell_dofs(cell->index());
    const ArrayView<const dolfin::la_index> dofs_L
      = dofmap_L->cell_dofs(cell->index());

    // Check that the local matrix is square
    if (dofs_a0.size() != dofs_a1.size())
    {
      dolfin_error("LocalSolver.cpp",
                   "assemble local LHS",
                   "Local LHS dimensions is non square (%d x %d) on cell %d",
                   dofs_a0.size(), dofs_a1.size(), cell->index());
    }

    // Check that the local RHS matches the LHS
    if (dofs_a0.size() != dofs_L.size())
    {
      dolfin_error("LocalSolver.cpp",
                   "assemble local RHS",
                   "Local RHS dimension %d is does not match first dimension "
                   "%d of LHS on cell %d",
                   dofs_L.size(), dofs_a0.size(), cell->index());
    }

    // Update data to current cell
    cell->get_coordinate_dofs(coordinate_dofs);

    // Assemble the linear form
    x_e.resize(dofs_L.size());
    b_e.resize(dofs_L.size(), 1);

    if (global_b)
    {
      // Copy global RHS data into local RHS vector
      global_b->get_local(b_e.data(), dofs_L.size(), dofs_L.data());
    }
    else
    {
      // Assemble local RHS vector
      LocalAssembler::assemble(b_e, *ufc_L, coordinate_dofs, ufc_cell,
                               *cell, cell_domains,
                               exterior_facet_domains, interior_facet_domains);
    }

    if (use_cache)
    {
      // Use cached factorisations
      if (_solver_type == SolverType::Cholesky)
        x_e = _cholesky_cache[cell->index()].solve(b_e);
      else
        x_e = _lu_cache[cell->index()].solve(b_e);
    }
    else
    {
      // Assemble the bilinear form
      A_e.resize(dofs_a0.size(), dofs_a1.size());
      LocalAssembler::assemble(A_e, ufc_a, coordinate_dofs,
                               ufc_cell, *cell, cell_domains,
                               exterior_facet_domains, interior_facet_domains);

      // Factorise and solve
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

    // Insert solution in global vector
    x.set_local(x_e.data(), dofs_a1.size(), dofs_a1.data());

    // Update progress
    p++;
  }

  // Finalise vector
  x.apply("insert");
}
//----------------------------------------------------------------------------
void LocalSolver::factorize()
{
  // Check that we have valid bilinear and linear forms
  dolfin_assert(_a);
  dolfin_assert(_a->rank() == 2);

  // Set timer
  Timer timer("Factorise local problems");

  // Extract the mesh
  dolfin_assert(_a->function_space(0)->mesh());
  const Mesh& mesh = *_a->function_space(0)->mesh();

  // Resize cache
  if (_solver_type == SolverType::Cholesky)
    _cholesky_cache.resize(mesh.num_cells());
  else
    _lu_cache.resize(mesh.num_cells());

  // Create UFC objects
  UFC ufc_a(*_a);

  // Get dofmaps
  std::array<std::shared_ptr<const GenericDofMap>, 2> dofmaps_a
    = {{_a->function_space(0)->dofmap(), _a->function_space(1)->dofmap()}};
  dolfin_assert(dofmaps_a[0] and dofmaps_a[1]);

  // Extract cell_domains etc from left-hand side form
  const MeshFunction<std::size_t>* cell_domains
    = _a->cell_domains().get();
  const MeshFunction<std::size_t>* exterior_facet_domains
    = _a->exterior_facet_domains().get();
  const MeshFunction<std::size_t>* interior_facet_domains
    = _a->interior_facet_domains().get();

  // Local dense matrix
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_e;

  // Loop over cells and solve local problems
  Progress p("Performing local (cell-wise) factorization", mesh.num_cells());
  ufc::cell ufc_cell;
  std::vector<double> coordinate_dofs;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get local-to-global dof maps for cell
    const ArrayView<const dolfin::la_index> dofs_a0
      = dofmaps_a[0]->cell_dofs(cell->index());
    const ArrayView<const dolfin::la_index> dofs_a1
      = dofmaps_a[1]->cell_dofs(cell->index());

    // Check that the local matrix is square
    if (dofs_a0.size() != dofs_a1.size())
    {
      dolfin_error("LocalSolver.cpp",
                   "assemble local LHS",
                   "Local LHS dimensions is non square (%d x %d) on cell %d",
                   dofs_a0.size(), dofs_a1.size(), cell->index());
    }

    // Update data to current cell
    cell->get_coordinate_dofs(coordinate_dofs);
    A_e.resize(dofs_a0.size(), dofs_a1.size());

    // Assemble the bilinear form
    LocalAssembler::assemble(A_e, ufc_a, coordinate_dofs,
                             ufc_cell, *cell, cell_domains,
                             exterior_facet_domains, interior_facet_domains);

    if (_solver_type == SolverType::Cholesky)
      _cholesky_cache[cell->index()].compute(A_e);
    else
      _lu_cache[cell->index()].compute(A_e);

    // Update progress
    p++;
  }
}
//----------------------------------------------------------------------------
void LocalSolver::clear_factorization()
{
  _cholesky_cache.clear();
  _lu_cache.clear();
}
//-----------------------------------------------------------------------------
