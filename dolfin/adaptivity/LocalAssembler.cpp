// Copyright (C) 2011 Marie E. Rognes
// Licensed under the GNU LGPL Version 3.0 or any later version
//
// First added:  2011-01-04
// Last changed: 2011-01-14

#include <dolfin/fem/UFC.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include "LocalAssembler.h"

using namespace dolfin;

//------------------------------------------------------------------------------
void LocalAssembler::assemble(arma::mat& A,
                              UFC& ufc,
                              const Cell& cell,
                              const MeshFunction<uint>* cell_domains,
                              const MeshFunction<uint>* exterior_facet_domains,
                              const MeshFunction<uint>* interior_facet_domains)
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
                                   const MeshFunction<uint>* domains)
{
  // Skip if there are no cell integrals
  if (ufc.form.num_cell_integrals() == 0)
    return;

  // Extract default cell integral
  ufc::cell_integral* integral = ufc.cell_integrals[0].get();

  // Get integral for sub domain (if any)
  if (domains && domains->size() > 0)
  {
    const uint domain = (*domains)[cell];
    if (domain < ufc.form.num_cell_integrals())
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
  integral->tabulate_tensor(ufc.A.get(), ufc.w, ufc.cell);

  // Stuff a_ufc.A into A
  const uint M = A.n_rows;
  const uint N = A.n_cols;
  for (uint i=0; i < M; i++)
    for (uint j=0; j < N; j++)
      A(i, j) += ufc.A[N*i + j];

}
//------------------------------------------------------------------------------
void LocalAssembler::assemble_exterior_facet(arma::mat& A,
                                             UFC& ufc,
                                             const Cell& cell,
                                             const Facet& facet,
                                             const uint local_facet,
                                             const MeshFunction<uint>* domains)
{
  // Skip if there are no exterior facet integrals
  if (ufc.form.num_exterior_facet_integrals() == 0)
    return;

  // Extract default exterior facet integral
  ufc::exterior_facet_integral* integral = ufc.exterior_facet_integrals[0].get();

  // Get integral for sub domain (if any)
  if (domains && domains->size() > 0)
  {
    const uint domain = (*domains)[facet];
    if (domain < ufc.form.num_exterior_facet_integrals())
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
  integral->tabulate_tensor(ufc.A.get(), ufc.w, ufc.cell, local_facet);

  // Stuff a_ufc.A into A
  const uint M = A.n_rows;
  const uint N = A.n_cols;
  for (uint i=0; i < M; i++)
  {
    for (uint j=0; j < N; j++)
      A(i, j) += ufc.A[N*i + j];
  }
}
//------------------------------------------------------------------------------
void LocalAssembler::assemble_interior_facet(arma::mat& A,
                                             UFC& ufc,
                                             const Cell& cell,
                                             const Facet& facet,
                                             const uint local_facet,
                                             const MeshFunction<uint>* domains)
{
  // Skip if there are no interior facet integrals
  if (ufc.form.num_interior_facet_integrals() == 0)
    return;

  // Extract default interior facet integral
  ufc::interior_facet_integral* integral = ufc.interior_facet_integrals[0].get();

  // Get integral for sub domain (if any)
  if (domains && domains->size() > 0)
  {
    const uint domain = (*domains)[facet];
    if (domain < ufc.form.num_interior_facet_integrals())
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
  integral->tabulate_tensor(ufc.macro_A.get(), ufc.macro_w,
                            ufc.cell0, ufc.cell1,
                            local_facet, local_facet);

  // Stuff upper left quadrant (corresponding to this cell) into A
  const uint M = A.n_rows;
  const uint N = A.n_cols;
  for (uint i=0; i < M; i++)
    for (uint j=0; j < N; j++)
      A(i, j) += ufc.macro_A[2*N*i + j]; // FIXME: row/col swap for vectors!
}
//------------------------------------------------------------------------------
