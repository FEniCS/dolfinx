// Copyright (C) 2011 Marie E. Rognes
// Licensed under the GNU LGPL Version 3.0 or any later version
//
// First added:  2011-01-04
// Last changed: 2011-01-05

#include <dolfin/fem/UFC.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include "LocalAssembler.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void LocalAssembler::assemble_cell(arma::mat& A, const uint N,
                                   UFC& ufc,
                                   const Cell& cell,
                                   std::vector<uint> exterior_facets,
                                   std::vector<uint> interior_facets)
{
  uint local_facet = 0;

  // Iterate over cell_integral(s) (assume max 1 for now)
  if (ufc.form.num_cell_integrals() == 1)
  {
    // Extract cell integral
    ufc::cell_integral* integral = ufc.cell_integrals[0].get();

    // Update ufc object to this cell
    ufc.update(cell);

    // Tabulate local tensor
    integral->tabulate_tensor(ufc.A.get(), ufc.w, ufc.cell);

    // Stuff a_ufc.A into A
    for (uint i=0; i < N; i++)
    {
      for (uint j=0; j < N; j++) {
        A(i, j) += ufc.A[N*i + j];
      }
    }
  }

  // Iterate over exterior_facet integral(s) (assume max 1 for now)
  if (ufc.form.num_exterior_facet_integrals() == 1)
  {
    // Extract exterior facet integral
    ufc::exterior_facet_integral* ef_integral = ufc.exterior_facet_integrals[0].get();

    // Iterate over given exterior facets
    for (uint k=0; k < exterior_facets.size(); k++)
    {
      // Get local index of facet with respect to the cell
      local_facet = exterior_facets[k];

      // Update to current cell and facet
      ufc.update(cell, local_facet);

      // Tabulate exterior facet tensor
      ef_integral->tabulate_tensor(ufc.A.get(), ufc.w, ufc.cell, local_facet);

      // Stuff a_ufc.A into A
      for (uint i=0; i < N; i++)
      {
        for (uint j=0; j < N; j++)
          A(i, j) += ufc.A[N*i + j];
      }
    }
  }

  // Iterate over interior facet integral(s) (assume max 1 for now)
  if (ufc.form.num_interior_facet_integrals() == 1)
  {
    // Extract exterior facet integral
    ufc::interior_facet_integral* if_integral = ufc.interior_facet_integrals[0].get();

    // Iterate over given interior facets
    for (uint k=0; k < interior_facets.size(); k++)
    {
      // Get local index of facet with respect to the cell
      local_facet = interior_facets[k];

      // Update to current pair of cells and facets
      ufc.update(cell, local_facet, cell, local_facet);

      // Tabulate interior facet tensor on macro element
      if_integral->tabulate_tensor(ufc.macro_A.get(), ufc.macro_w,
                                   ufc.cell0, ufc.cell1,
                                   local_facet, local_facet);

      // Stuff a_ufc.A into A
      for (uint i=0; i < N; i++)
      {
        for (uint j=0; j < N; j++)
          A(i, j) += ufc.macro_A[2*N*i + j];
      }
    }
  }
}
//-----------------------------------------------------------------------------
void LocalAssembler::assemble_cell(arma::vec& b, const uint N,
                                   UFC& ufc,
                                   const Cell& cell,
                                   std::vector<uint> exterior_facets,
                                   std::vector<uint> interior_facets)
{
  // Iterate over cell_integral(s) (assume max 1 for now)
  if (ufc.form.num_cell_integrals() == 1)
  {
    // Extract cell integral
    ufc::cell_integral* integral = ufc.cell_integrals[0].get();

    // Update ufc object to this cell
    ufc.update(cell);

    // Tabulate local tensor
    integral->tabulate_tensor(ufc.A.get(), ufc.w, ufc.cell);

    // Stuff a_ufc.A into b
    for (uint i=0; i < N; i++)
      b(i) += ufc.A[i];
  }

  // Iterate over exterior_facet integral(s) (assume max 1 for now)
  if (ufc.form.num_exterior_facet_integrals() == 1)
  {
    // Extract exterior facet integral
    ufc::exterior_facet_integral* ef_integral = ufc.exterior_facet_integrals[0].get();

    // Iterate over given exterior facets
    for (uint k=0; k < exterior_facets.size(); k++)
    {
      // Get local index of facet with respect to the cell
      const uint local_facet = exterior_facets[k];

      // Update to current cell and facet
      ufc.update(cell, local_facet);

      // Tabulate exterior facet tensor
      ef_integral->tabulate_tensor(ufc.A.get(), ufc.w, ufc.cell, local_facet);

      // Stuff a_ufc.A into A
      for (uint i=0; i < N; i++)
        b(i) += ufc.A[i];
    }
  }

  // Iterate over interior facet integral(s) (assume max 1 for now)
  if (ufc.form.num_interior_facet_integrals() == 1)
  {
    // Extract interior facet integral
    ufc::interior_facet_integral* if_integral = ufc.interior_facet_integrals[0].get();

    // Iterate over given exterior facets
    for (uint k=0; k < interior_facets.size(); k++)
    {
      // Get local index of facet with respect to the cell
      const uint local_facet = interior_facets[k];

      // Update to current pair of cells and facets
      ufc.update(cell, local_facet, cell, local_facet);

      // Tabulate exterior interior facet tensor on macro element
      if_integral->tabulate_tensor(ufc.macro_A.get(), ufc.macro_w,
                                ufc.cell0, ufc.cell1,
                                local_facet, local_facet);

      // Stuff correct pieces from a_ufc.A into b
      for (uint i=0; i < N; i++)
        b(i) += ufc.macro_A[i];
    }
  }
}
//-----------------------------------------------------------------------------
