// Copyright (C) 2009-2011 Anders Logg
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
// Modified by Marie E. Rognes, 2010
// Modified by Garth N. Wells, 2010
//
// First added:  2009-12-08
// Last changed: 2011-11-12
//

#include <vector>
#include <armadillo>

#include <dolfin/common/Array.h>
#include <dolfin/common/Timer.h>
#include <dolfin/fem/BasisFunction.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/FacetCell.h>
#include "Extrapolation.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void Extrapolation::extrapolate(Function& w, const Function& v)
{
  // Using set_local for simplicity here
  not_working_in_parallel("Extrapolation");

  // Too verbose
  //info("Extrapolating function: %s --> %s",
  //     v.function_space().element().signature().c_str(),
  //     w.function_space().element().signature().c_str());

  // Check that the meshes are the same
  if (w.function_space()->mesh() != v.function_space()->mesh())
  {
    dolfin_error("Extrapolation.cpp",
                 "compute extrapolation",
                 "Extrapolation must be computed on the same mesh");
  }

  // Extract mesh and function spaces
  const FunctionSpace& V = *v.function_space();
  const FunctionSpace& W = *w.function_space();
  dolfin_assert(V.mesh());
  const Mesh& mesh = *V.mesh();

  // Initialize cell-cell connectivity
  const uint D = mesh.topology().dim();
  mesh.init(D, D);

  // UFC cell view of center cell
  UFCCell c0(mesh);

  // List of values for each dof of w (multivalued until we average)
  std::vector<std::vector<double> > coefficients;
  coefficients.resize(W.dim());

  // Iterate over cells in mesh
  dolfin_assert(W.dofmap());
  for (CellIterator cell0(mesh); !cell0.end(); ++cell0)
  {
    // Update UFC view
    c0.update(*cell0);

    // Tabulate dofs for w on cell and store values
    const std::vector<std::size_t>& dofs = W.dofmap()->cell_dofs(cell0->index());

    // Compute coefficients on this cell
    std::size_t offset = 0;
    compute_coefficients(coefficients, v, V, W, *cell0, c0, dofs, offset);
  }

  // Average coefficients
  average_coefficients(w, coefficients);
}
//-----------------------------------------------------------------------------
void Extrapolation::compute_coefficients(std::vector<std::vector<double> >& coefficients,
                                         const Function& v,
                                         const FunctionSpace& V,
                                         const FunctionSpace& W,
                                         const Cell& cell0,
                                         const ufc::cell& c0,
                                         const std::vector<std::size_t>& dofs,
                                         std::size_t& offset)
{
  // Call recursively for mixed elements
  dolfin_assert(V.element());
  const uint num_sub_spaces = V.element()->num_sub_elements();
  if (num_sub_spaces > 0)
  {
    for (uint k = 0; k < num_sub_spaces; k++)
    {
      compute_coefficients(coefficients, v[k], *V[k], *W[k], cell0, c0,
                           dofs, offset);
    }
    return;
  }

  // Build data structures for keeping track of unique dofs
  std::map<std::size_t, std::map<std::size_t, std::size_t> > cell2dof2row;
  std::set<std::size_t> unique_dofs;
  build_unique_dofs(unique_dofs, cell2dof2row, cell0, c0, V);

  // Compute size of linear system
  dolfin_assert(W.element());
  const std::size_t N = W.element()->space_dimension();
  const std::size_t M = unique_dofs.size();

  // Check size of system
  if (M < N)
  {
    dolfin_error("Extrapolation.cpp",
                 "compute extrapolation",
                 "Not enough degrees of freedom on local patch to build extrapolation");
  }

  // Create matrix and vector for linear system
  arma::mat A(M, N);
  arma::vec b(M);

  // Add equations on cell and neighboring cells
  add_cell_equations(A, b, cell0, cell0, c0, c0, V, W, v, cell2dof2row[cell0.index()]);
  dolfin_assert(V.mesh());
  UFCCell c1(*V.mesh());
  for (CellIterator cell1(cell0); !cell1.end(); ++cell1)
  {
    if (cell2dof2row[cell1->index()].empty())
      continue;

    c1.update(*cell1);
    add_cell_equations(A, b, cell0, *cell1, c0, c1, V, W, v, cell2dof2row[cell1->index()]);
  }

  // Solve least squares system
  arma::Col<double> x = arma::solve(A, b);

  // Insert resulting coefficients into global coefficient vector
  dolfin_assert(W.dofmap());
  for (uint i = 0; i < W.dofmap()->cell_dimension(cell0.index()); ++i)
    coefficients[dofs[i + offset]].push_back(x[i]);

  // Increase offset
  offset += W.dofmap()->cell_dimension(cell0.index());
}
//-----------------------------------------------------------------------------
void Extrapolation::build_unique_dofs(std::set<std::size_t>& unique_dofs,
                                      std::map<std::size_t, std::map<std::size_t, std::size_t> >& cell2dof2row,
                                      const Cell& cell0,
                                      const ufc::cell& c0,
                                      const FunctionSpace& V)
{
  // Counter for matrix row index
  std::size_t row = 0;
  dolfin_assert(V.mesh());
  UFCCell c1(*V.mesh());

  // Compute unique dofs on center cell
  cell2dof2row[cell0.index()] = compute_unique_dofs(cell0, c0, V, row, unique_dofs);

  // Compute unique dofs on neighbouring cells
  for (CellIterator cell1(cell0); !cell1.end(); ++cell1)
  {
    c1.update(*cell1);
    cell2dof2row[cell1->index()] = compute_unique_dofs(*cell1, c1, V, row, unique_dofs);
  }
}
//-----------------------------------------------------------------------------
void Extrapolation::add_cell_equations(arma::Mat<double>& A,
                                       arma::Col<double>& b,
                                       const Cell& cell0,
                                       const Cell& cell1,
                                       const ufc::cell& c0,
                                       const ufc::cell& c1,
                                       const FunctionSpace& V,
                                       const FunctionSpace& W,
                                       const Function& v,
                                       std::map<std::size_t, std::size_t>& dof2row)
{
  // Extract coefficents for v on patch cell
  dolfin_assert(V.element());
  std::vector<double> dof_values(V.element()->space_dimension());
  v.restrict(&dof_values[0], *V.element(), cell1, c1);

  // Iterate over given local dofs for V on patch cell
  dolfin_assert(W.element());
  for (std::map<std::size_t, std::size_t>::iterator it = dof2row.begin(); it!= dof2row.end(); it++)
  {
    const std::size_t i = it->first;
    const std::size_t row = it->second;

    // Iterate over basis functions for W on center cell
    for (std::size_t j = 0; j < W.element()->space_dimension(); ++j)
    {

      // Create basis function
      const BasisFunction phi(j, *W.element(), c0);

      // Evaluate dof on basis function
      const double dof_value = V.element()->evaluate_dof(i, phi, c1);

      // Insert dof_value into matrix
      A(row, j) = dof_value;
    }

    // Insert coefficient into vector
    b[row] = dof_values[i];
  }
}
//-----------------------------------------------------------------------------
std::map<std::size_t, std::size_t>
Extrapolation::compute_unique_dofs(const Cell& cell, const ufc::cell& c,
                                   const FunctionSpace& V,
                                   std::size_t& row,
                                   std::set<std::size_t>& unique_dofs)
{
  dolfin_assert(V.dofmap());
  const std::vector<std::size_t>& dofs = V.dofmap()->cell_dofs(cell.index());

  // Data structure for current cell
  std::map<std::size_t, std::size_t> dof2row;

  for (uint i = 0; i < V.dofmap()->cell_dimension(cell.index()); ++i)
  {
    // Ignore if this degree of freedom is already considered
    if (unique_dofs.find(dofs[i]) != unique_dofs.end())
      continue;

    // Put global index into unique_dofs
    unique_dofs.insert(dofs[i]);

    // Map local dof index to current matrix row-index
    dof2row[i] = row;

    // Increase row index
    row++;
  }

  return dof2row;
}
//-----------------------------------------------------------------------------
void Extrapolation::average_coefficients(Function& w,
                               std::vector<std::vector<double> >& coefficients)
{
  const FunctionSpace& W = *w.function_space();
  std::vector<double> dof_values(W.dim());

  for (std::size_t i = 0; i < W.dim(); i++)
  {
    double s = 0.0;
    for (uint j = 0; j < coefficients[i].size(); ++j)
      s += coefficients[i][j];

    s /= static_cast<double>(coefficients[i].size());
    dof_values[i] = s;
  }

  // Update dofs for w
  dolfin_assert(w.vector());
  w.vector()->set_local(dof_values);
}
//-----------------------------------------------------------------------------
