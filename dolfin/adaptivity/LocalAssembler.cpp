// Copyright (C) 2011 Marie E. Rognes
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
// Modified by Anders Logg 2013
//
// First added:  2011-01-04
// Last changed: 2013-01-10

#include <Eigen/Dense>
#include <dolfin/fem/UFC.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include "LocalAssembler.h"

using namespace dolfin;

//------------------------------------------------------------------------------
void
LocalAssembler::assemble(Eigen::MatrixXd& A,
                         UFC& ufc,
                         ufc::cell& ufc_cell,
                         const Cell& cell,
                         const MeshFunction<std::size_t>* cell_domains,
                         const MeshFunction<std::size_t>* exterior_facet_domains,
                         const MeshFunction<std::size_t>* interior_facet_domains)
{
  // Clear tensor
  A.setZero();

  cell.ufc_cell_geometry(ufc_cell);

  // Assemble contributions from cell integral
  assemble_cell(A, ufc, ufc_cell, cell, cell_domains);

  // Assemble contributions from facet integrals
  for (FacetIterator facet(cell); !facet.end(); ++facet)
  {
    ufc_cell.local_facet = facet.pos();
    if (facet->num_entities(cell.dim()) == 2)
    {
      assemble_interior_facet(A, ufc, ufc_cell, cell, *facet,
                              facet.pos(), interior_facet_domains);
    }
    else
    {
      assemble_exterior_facet(A, ufc, ufc_cell, cell, *facet,
                              facet.pos(), exterior_facet_domains);
    }
  }
}
//------------------------------------------------------------------------------
void LocalAssembler::assemble_cell(Eigen::MatrixXd& A,
                                   UFC& ufc,
                                   const ufc::cell& ufc_cell,
                                   const Cell& cell,
                                   const MeshFunction<std::size_t>* domains)
{
  // Skip if there are no cell integrals
  if (!ufc.form.has_cell_integrals())
    return;

  // Extract default cell integral
  ufc::cell_integral* integral = ufc.default_cell_integral.get();

  // Get integral for sub domain (if any)
  if (domains && !domains->empty())
    integral = ufc.get_cell_integral((*domains)[cell]);

  // Skip integral if zero
  if (!integral)
    return;

  // Update to current cell
  ufc.update(cell, ufc_cell);

  // Tabulate cell tensor
  integral->tabulate_tensor(ufc.A.data(),
                            ufc.w(),
                            ufc_cell.vertex_coordinates.data(),
                            ufc_cell.orientation);

  // Stuff a_ufc.A into A
  const std::size_t M = A.rows();
  const std::size_t N = A.cols();
  for (std::size_t i = 0; i < M; i++)
    for (std::size_t j = 0; j < N; j++)
      A(i, j) += ufc.A[N*i + j];
}
//------------------------------------------------------------------------------
void
LocalAssembler::assemble_exterior_facet(Eigen::MatrixXd& A,
                                        UFC& ufc,
                                        const ufc::cell& ufc_cell,
                                        const Cell& cell,
                                        const Facet& facet,
                                        const std::size_t local_facet,
                                        const MeshFunction<std::size_t>* domains)
{
  // Skip if there are no exterior facet integrals
  if (!ufc.form.has_exterior_facet_integrals())
    return;

  // Extract default exterior facet integral
  ufc::exterior_facet_integral* integral
    = ufc.default_exterior_facet_integral.get();

  // Get integral for sub domain (if any)
  if (domains && !domains->empty())
    integral = ufc.get_exterior_facet_integral((*domains)[facet]);

  // Skip integral if zero
  if (!integral)
    return;

  // Update to current cell
  ufc.update(cell, ufc_cell);

  // Tabulate exterior facet tensor
  integral->tabulate_tensor(ufc.A.data(),
                            ufc.w(),
                            ufc_cell.vertex_coordinates.data(),
                            local_facet);

  // Stuff a_ufc.A into A
  const std::size_t M = A.rows();
  const std::size_t N = A.cols();
  for (std::size_t i = 0; i < M; i++)
    for (std::size_t j = 0; j < N; j++)
      A(i, j) += ufc.A[N*i + j];
}
//------------------------------------------------------------------------------
void
LocalAssembler::assemble_interior_facet(Eigen::MatrixXd& A,
                                        UFC& ufc,
                                        const ufc::cell& ufc_cell,
                                        const Cell& cell,
                                        const Facet& facet,
                                        const std::size_t local_facet,
                                        const MeshFunction<std::size_t>* domains)
{
  // Skip if there are no interior facet integrals
  if (!ufc.form.has_interior_facet_integrals())
    return;

  // Extract default interior facet integral
  ufc::interior_facet_integral* integral
    = ufc.default_interior_facet_integral.get();

  // Get integral for sub domain (if any)
  if (domains && !domains->empty())
    integral = ufc.get_interior_facet_integral((*domains)[facet]);

  // Skip integral if zero
  if (!integral)
    return;

  // Update to current pair of cells and facets
  ufc.update(cell, ufc_cell, cell, ufc_cell);

  // Tabulate interior facet tensor on macro element
  integral->tabulate_tensor(ufc.macro_A.data(), ufc.macro_w(),
                            ufc_cell.vertex_coordinates.data(),
                            ufc_cell.vertex_coordinates.data(),
                            local_facet, local_facet);

  // Stuff upper left quadrant (corresponding to this cell) into A
  const std::size_t M = A.rows();
  const std::size_t N = A.cols();
  if (N == 1)
  {
    for (std::size_t i = 0; i < M; i++)
      A(i, 0) = ufc.macro_A[i];
  }
  else
  {
    for (std::size_t i = 0; i < M; i++)
      for (std::size_t j = 0; j < N; j++)
        A(i, j) += ufc.macro_A[2*N*i + j];
  }
}
//------------------------------------------------------------------------------
