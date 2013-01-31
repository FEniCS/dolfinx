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

#include <armadillo>
#include <dolfin/fem/UFC.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include "LocalAssembler.h"

using namespace dolfin;

//------------------------------------------------------------------------------
void LocalAssembler::assemble(arma::Mat<double>& A,
                              UFC& ufc,
                              const Cell& cell,
                              const MeshFunction<std::size_t>* cell_domains,
                              const MeshFunction<std::size_t>* exterior_facet_domains,
                              const MeshFunction<std::size_t>* interior_facet_domains)
{
  // Clear tensor
  A.zeros();

  // Assemble contributions from cell integral
  assemble_cell(A, ufc, cell, cell_domains);

  // Assemble contributions from facet integrals
  for (FacetIterator facet(cell); !facet.end(); ++facet)
  {
    if (facet->num_entities(cell.dim()) == 2)
      assemble_interior_facet(A, ufc, cell, *facet,
                              facet.pos(), interior_facet_domains);
    else
      assemble_exterior_facet(A, ufc, cell, *facet,
                              facet.pos(), exterior_facet_domains);
  }
}
//------------------------------------------------------------------------------
void LocalAssembler::assemble_cell(arma::mat& A,
                                   UFC& ufc,
                                   const Cell& cell,
                                   const MeshFunction<std::size_t>* domains)
{
  // Skip if there are no cell integrals
  if (ufc.form.num_cell_domains() == 0)
    return;

  // Extract default cell integral
  ufc::cell_integral* integral = ufc.cell_integrals[0].get();

  // Get integral for sub domain (if any)
  if (domains && !domains->empty())
  {
    const std::size_t domain = (*domains)[cell];
    if (domain < ufc.form.num_cell_domains())
      integral = ufc.cell_integrals[domain].get();
    else
      return;
  }

  // Skip integral if zero
  if (!integral)
    return;

  // Update to current cell
  ufc.update(cell);

  // Tabulate cell tensor
  integral->tabulate_tensor(&ufc.A[0], ufc.w(), &ufc.cell.vertex_coordinates[0]);

  // Stuff a_ufc.A into A
  const std::size_t M = A.n_rows;
  const std::size_t N = A.n_cols;
  for (std::size_t i = 0; i < M; i++)
    for (std::size_t j = 0; j < N; j++)
      A(i, j) += ufc.A[N*i + j];
}
//------------------------------------------------------------------------------
void LocalAssembler::assemble_exterior_facet(arma::mat& A,
                                             UFC& ufc,
                                             const Cell& cell,
                                             const Facet& facet,
                                             const std::size_t local_facet,
                                             const MeshFunction<std::size_t>* domains)
{
  // Skip if there are no exterior facet integrals
  if (ufc.form.num_exterior_facet_domains() == 0)
    return;

  // Extract default exterior facet integral
  ufc::exterior_facet_integral* integral = ufc.exterior_facet_integrals[0].get();

  // Get integral for sub domain (if any)
  if (domains && !domains->empty())
  {
    const std::size_t domain = (*domains)[facet];
    if (domain < ufc.form.num_exterior_facet_domains())
      integral = ufc.exterior_facet_integrals[domain].get();
    else
      return;
  }

  // Skip integral if zero
  if (!integral)
    return;

  // Update to current cell
  ufc.update(cell, local_facet);

  // Tabulate exterior facet tensor
  integral->tabulate_tensor(&ufc.A[0],
                            ufc.w(),
                            &ufc.cell.vertex_coordinates[0],
                            local_facet);

  // Stuff a_ufc.A into A
  const std::size_t M = A.n_rows;
  const std::size_t N = A.n_cols;
  for (std::size_t i=0; i < M; i++)
    for (std::size_t j=0; j < N; j++)
      A(i, j) += ufc.A[N*i + j];
}
//------------------------------------------------------------------------------
void LocalAssembler::assemble_interior_facet(arma::mat& A,
                                             UFC& ufc,
                                             const Cell& cell,
                                             const Facet& facet,
                                             const std::size_t local_facet,
                                             const MeshFunction<std::size_t>* domains)
{
  // Skip if there are no interior facet integrals
  if (ufc.form.num_interior_facet_domains() == 0)
    return;

  // Extract default interior facet integral
  ufc::interior_facet_integral* integral = ufc.interior_facet_integrals[0].get();

  // Get integral for sub domain (if any)
  if (domains && !domains->empty())
  {
    const std::size_t domain = (*domains)[facet];
    if (domain < ufc.form.num_interior_facet_domains())
      integral = ufc.interior_facet_integrals[domain].get();
    else
      return;
  }

  // Skip integral if zero
  if (!integral)
    return;

  // Update to current pair of cells and facets
  ufc.update(cell, local_facet, cell, local_facet);

  // Tabulate interior facet tensor on macro element
  integral->tabulate_tensor(&ufc.macro_A[0], ufc.macro_w(),
                            &ufc.cell0.vertex_coordinates[0],
                            &ufc.cell1.vertex_coordinates[0],
                            local_facet, local_facet);

  // Stuff upper left quadrant (corresponding to this cell) into A
  const std::size_t M = A.n_rows;
  const std::size_t N = A.n_cols;

  if (N == 1)
    for (std::size_t i=0; i < M; i++)
      A(i, 0) = ufc.macro_A[i];
  else
    for (std::size_t i=0; i < M; i++)
      for (std::size_t j=0; j < N; j++)
        A(i, j) += ufc.macro_A[2*N*i + j];
}
//------------------------------------------------------------------------------
