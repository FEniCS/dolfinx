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
#include <Eigen/Dense>

#include <dolfin/la/GenericVector.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/common/Timer.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/la/GenericLinearAlgebraFactory.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/Function.h>
#include "assemble.h"
#include "GenericDofMap.h"
#include "Form.h"
#include "UFC.h"

#include "LocalSolver.h"

using namespace dolfin;

//----------------------------------------------------------------------------
LocalSolver::LocalSolver(std::shared_ptr<const Form> a,
                         std::shared_ptr<const Form> L) : _a(a), _L(L)
{
  dolfin_assert(a);
  dolfin_assert(a->rank() == 2);
  dolfin_assert(L);
  dolfin_assert(L->rank() == 1);
}
//----------------------------------------------------------------------------
/*
LocalSolver::LocalSolver(std::shared_ptr<const Form> a) : _a(a)
{
  init();
}
*/
//----------------------------------------------------------------------------
void LocalSolver::solve_global_rhs(GenericVector& x)
{
  dolfin_assert(_a);
  dolfin_assert(_a->rank() == 2);
  dolfin_assert(_L->rank() == 1);

  // FIXME: check that problem is square

  // Create UFC objects
  UFC ufc_a(*this->_a);
  UFC ufc_L(*this->_L);

  // Set timer
  Timer timer("Local solver");

  // Extract mesh
  dolfin_assert(_a->function_space(0)->mesh());
  const Mesh& mesh = *_a->function_space(0)->mesh();

  // Get cell integral for RHS
  ufc::cell_integral* integral_a = ufc_a.default_cell_integral.get();
  dolfin_assert(integral_a);

  // Get dofmaps
  std::array<std::shared_ptr<const GenericDofMap>, 2> dofmaps_a
             = {_a->function_space(0)->dofmap(),
                _a->function_space(1)->dofmap()};
  dolfin_assert(dofmaps_a[0] and dofmaps_a[1]);
  dolfin_assert(_L->function_space(0)->dofmap());
  const GenericDofMap& dofmap_L = *_L->function_space(0)->dofmap();

  // FIXME: initialise properly with ghosts and local-to-global map
  // Initialise vector
  if (x.empty())
  {
    const std::pair<std::size_t, std::size_t> local_range
      = dofmap_L.ownership_range();
    x.init(mesh.mpi_comm(), local_range);
  }

  // Assemble global RHS vector
  std::shared_ptr<GenericVector> b = x.factory().create_vector();
  dolfin_assert(b);
  assemble(*b, *_L);

  // Eigen data structures
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_e;
  Eigen::VectorXd b_e, x_e;;
  Eigen::PartialPivLU<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>> lu;

  // Assemble LHS over cells and solve
  Progress p("Performing local (cell-wise) solve", mesh.num_cells());
  ufc::cell ufc_cell;
  std::vector<double> vertex_coordinates;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    cell->get_vertex_coordinates(vertex_coordinates);
    cell->get_cell_data(ufc_cell);

    // Get dofmaps
    const std::vector<dolfin::la_index>& dofs_L
      = dofmap_L.cell_dofs(cell->index());
    const std::vector<dolfin::la_index>& dofs_a1
      = dofmaps_a[0]->cell_dofs(cell->index());

    // Resize local RHS vector
    b_e.resize(dofmap_L.cell_dimension(cell->index()));

    // Copy global RHS data into local RHS
    b->get_local(b_e.data(), dofs_L.size(), dofs_L.data());

    // Solve local problem
    if (_lus.empty())
    {
      // Update UFC object
      ufc_a.update(*cell, vertex_coordinates, ufc_cell,
                   integral_a->enabled_coefficients());

      // Resize A
      const std::size_t dim = dofmaps_a[0]->cell_dimension(cell->index());
      A_e.resize(dim, dim);

      // Tabulate matrix on cell
      integral_a->tabulate_tensor(A_e.data(), ufc_a.w(),
                                  vertex_coordinates.data(),
                                  ufc_cell.orientation);
      lu.compute(A_e);
      x_e = lu.solve(b_e);
    }
    else
    {
      dolfin_assert(cell->index() < _lus.size());
      x_e = _lus[cell->index()].solve(b_e);
    }

    // Set solution in global vector
    x.set_local(x_e.data(), dofs_a1.size(), dofs_a1.data());

    p++;
  }
  // Finalise vector
  x.apply("insert");
}
//----------------------------------------------------------------------------
void LocalSolver::solve_local_rhs(GenericVector& x)
{
  dolfin_assert(_a);
  dolfin_assert(_a->rank() == 2);
  dolfin_assert(_L->rank() == 1);

  // FIXME: check that problem is square

  // Create UFC objects
  UFC ufc_a(*this->_a);
  UFC ufc_L(*this->_L);

  // Set timer
  Timer timer("Local solver");

  // Extract mesh
  dolfin_assert(_a->function_space(0)->mesh());
  const Mesh& mesh = *_a->function_space(0)->mesh();

  // Get cell integral for LHS and RHS
  ufc::cell_integral* integral_L = ufc_L.default_cell_integral.get();
  dolfin_assert(integral_L);
  ufc::cell_integral* integral_a = ufc_a.default_cell_integral.get();
  dolfin_assert(integral_a);

  // Get dofmaps
  std::array<std::shared_ptr<const GenericDofMap>, 2> dofmaps_a
             = {_a->function_space(0)->dofmap(),
                _a->function_space(1)->dofmap()};
  dolfin_assert(dofmaps_a[0] and dofmaps_a[1]);
  //dolfin_assert(_L->function_space(0)->dofmap());
  //const GenericDofMap& dofmap_L = *_L->function_space(0)->dofmap();

  // FIXME: initialise properly with ghosts and local-to-global map
  // Initialise vector
  if (x.empty())
  {
    dolfin_assert(_L->function_space(0)->dofmap());
    const GenericDofMap& dofmap_L = *_L->function_space(0)->dofmap();

    const std::pair<std::size_t, std::size_t> local_range
      = dofmap_L.ownership_range();
    x.init(mesh.mpi_comm(), local_range);
  }

  // Eigen data structures
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A_e;
  Eigen::VectorXd b_e, x_e;;
  Eigen::PartialPivLU<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>> lu;

  // Assemble LHS over cells and solve
  Progress p("Performing local (cell-wise) solve", mesh.num_cells());
  ufc::cell ufc_cell;
  std::vector<double> vertex_coordinates;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    cell->get_vertex_coordinates(vertex_coordinates);
    cell->get_cell_data(ufc_cell);

    // Get dofmaps
    //const std::vector<dolfin::la_index>& dofs_L
    //  = dofmap_L.cell_dofs(cell->index());
    const std::vector<dolfin::la_index>& dofs_a0
      = dofmaps_a[0]->cell_dofs(cell->index());

    // Resize local RHS vector
    b_e.resize(dofs_a0.size());

    // Update UFC object
    ufc_L.update(*cell, vertex_coordinates, ufc_cell,
                 integral_L->enabled_coefficients());

    // Tabulate matrix on cell
    integral_L->tabulate_tensor(b_e.data(), ufc_L.w(),
                                vertex_coordinates.data(),
                                ufc_cell.orientation);

    // Solve local problem
    if (_lus.empty())
    {
      // Update UFC object
      ufc_a.update(*cell, vertex_coordinates, ufc_cell,
                   integral_a->enabled_coefficients());

      // Resize A
      const std::size_t dim = dofmaps_a[0]->cell_dimension(cell->index());
      A_e.resize(dim, dim);

      // Tabulate matrix on cell
      integral_a->tabulate_tensor(A_e.data(), ufc_a.w(),
                                  vertex_coordinates.data(),
                                  ufc_cell.orientation);
      lu.compute(A_e);
      x_e = lu.solve(b_e);
    }
    else
    {
      dolfin_assert(cell->index() < _lus.size());
      x_e = _lus[cell->index()].solve(b_e);
    }

    // Set solution in global vector
    x.set_local(x_e.data(), dofs_a0.size(), dofs_a0.data());

    p++;
  }

  // Finalise vector
  x.apply("insert");
}
//----------------------------------------------------------------------------
/*
void LocalSolver::solve(GenericVector& x, const GenericVector& b) const
{
  UFC ufc_a(*this->_a);
  //UFC ufc_L(*this->_l);
  std::shared_ptr<const GenericVector> _x(reference_to_no_delete_pointer(x));

  // Set timer
  Timer timer("Local solver");

  // Extract mesh
  const Mesh& mesh = (*this->_a).mesh();

  // Form ranks
  const std::size_t rank_a = ufc_a.form.rank();
  //const std::size_t rank_L = ufc_L.form.rank();

  // Check form ranks
  dolfin_assert(rank_a == 2);
  //dolfin_assert(rank_L == 1);

  // Collect pointers to dof maps
  std::shared_ptr<const GenericDofMap> dofmap_a0
    = (*this->_a).function_space(0)->dofmap();
  std::shared_ptr<const GenericDofMap> dofmap_a1
    = (*this->_a).function_space(1)->dofmap();
  //std::shared_ptr<const GenericDofMap> dofmap_L
    //= (*this->_a).function_space(0)->dofmap();
  dolfin_assert(dofmap_a0);
  dolfin_assert(dofmap_a1);
  //dolfin_assert(dofmap_L);

  // Initialise vector
  if (x.empty())
  {
    std::pair<std::size_t, std::size_t> local_range
      = b.local_range();
    x.init(mesh.mpi_comm(), local_range);
  }

  // Eigen data structures
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A;
  Eigen::VectorXd b_local, x_local;;

  // Globally assemble RHS
  //if (this->_l->ufc_form())
    //assemble(*Gb, *this->_l);

  // Assemble over cells
  Progress p("Performing local (cell-wise) solve", mesh.num_cells());
  ufc::cell ufc_cell;
  std::vector<double> vertex_coordinates;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    cell->get_vertex_coordinates(vertex_coordinates);
    cell->get_cell_data(ufc_cell);

    // Get local-to-global dof maps for cell
    const std::vector<dolfin::la_index>& dofs_a0
      = dofmap_a0->cell_dofs(cell->index());
    //const std::vector<dolfin::la_index>& dofs_L
      //= dofmap_L->cell_dofs(cell->index());

    // Resize b
    b_local.resize(dofs_a0.size());

    // Extract local RHS info from global RHS
    b.get_local(b_local.data(), dofs_a0.size(), dofs_a0.data());

     // Solve local problem
    x_local = _lus[cell->index()].solve(b_local);

    // Set solution in global vector
    x.set_local(x_local.data(), dofs_a0.size(), dofs_a0.data());

    p++;
  }
  // Finalise vector
  x.apply("insert");
}
*/
//----------------------------------------------------------------------------
void LocalSolver::clear_factorization()
{
  _lus.clear();
}
//-----------------------------------------------------------------------------
void LocalSolver::factorize()
{
  // Create UFC object
  dolfin_assert(_a);
  UFC ufc(*_a);

  // Check rank
  dolfin_assert(ufc.form.rank() == 2);

  // Raise error for Point integrals
  if (ufc.form.has_point_integrals())
  {
    dolfin_error("LocalSolver.cpp",
                 "assemble system",
                 "Point integrals are not supported (yet)");
  }

  // Set timer
  Timer timer("Factorize local block matrices");

  // Extract mesh
  const Mesh& mesh = _a->mesh();

  // Collect pointers to dof maps
  std::array<std::shared_ptr<const GenericDofMap>, 2>
    dofmaps = {_a->function_space(0)->dofmap(),
               _a->function_space(1)->dofmap()};
  dolfin_assert(dofmaps[0]);
  dolfin_assert(dofmaps[1]);

  // Get cell integral
  ufc::cell_integral* integral = ufc.default_cell_integral.get();
  dolfin_assert(integral);

  // Resize LU cache
  _lus.resize(mesh.num_cells());

  // Eigen data structure for cell matrix
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A;

  // Assemble over cells
  Progress p("Performing local (cell-wise) solve", mesh.num_cells());
  ufc::cell ufc_cell;
  std::vector<double> vertex_coordinates;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    cell->get_vertex_coordinates(vertex_coordinates);
    cell->get_cell_data(ufc_cell);
    ufc.update(*cell, vertex_coordinates, ufc_cell,
               integral->enabled_coefficients());

    // Get cell dimension
    std::size_t dim = dofmaps[0]->cell_dimension(cell->index());

    // Check that local problem is square
    dolfin_assert(dim == dofmaps[1]->cell_dimension(cell->index()));

    // Resize element matrix
    A.resize(dim, dim);

    // Tabulate A on cell
    integral->tabulate_tensor(A.data(), ufc.w(),
                              vertex_coordinates.data(),
                              ufc_cell.orientation);

     // Compute LU decomposition and store
    _lus[cell->index()].compute(A);

    p++;
  }

}
//-----------------------------------------------------------------------------
