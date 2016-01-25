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
// Modified by Tormod Landet 2015
//
// First added:  2011-01-04
// Last changed: 2015-09-30

#include <dolfin/common/types.h>
#include <Eigen/Dense>

#include <dolfin/common/Array.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/UFC.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include "LocalAssembler.h"

using namespace dolfin;

//------------------------------------------------------------------------------
void
LocalAssembler::assemble(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>& A,
                         UFC& ufc,
                         const std::vector<double>& coordinate_dofs,
                         ufc::cell& ufc_cell,
                         const Cell& cell,
                         const MeshFunction<std::size_t>* cell_domains,
                         const MeshFunction<std::size_t>* exterior_facet_domains,
                         const MeshFunction<std::size_t>* interior_facet_domains)
{
  cell.get_cell_data(ufc_cell);

  // Assemble contributions from cell integral
  assemble_cell(A, ufc, coordinate_dofs, ufc_cell, cell, cell_domains);

  // Assemble contributions from facet integrals
  if (ufc.form.has_exterior_facet_integrals()
      || ufc.form.has_interior_facet_integrals())
  {
    for (FacetIterator facet(cell); !facet.end(); ++facet)
    {
      ufc_cell.local_facet = facet.pos();
      const int Ncells = facet->num_entities(cell.dim());
      if (Ncells == 2)
      {
        assemble_interior_facet(A, ufc, coordinate_dofs, ufc_cell, cell,
                                *facet, facet.pos(), interior_facet_domains,
                                cell_domains);
      }
      else if (Ncells == 1)
      {
        assemble_exterior_facet(A, ufc, coordinate_dofs, ufc_cell, cell,
                                *facet, facet.pos(), exterior_facet_domains);
      }
      else
      {
        dolfin_error("LocalAssembler.cpp",
                     "assemble local problem",
                     "Cell <-> facet connectivity not initialized, found "
                     "facet with %d connected cells. Expected 1 or 2 cells",
                     Ncells);
      }
    }
  }

  // Check that there are no vertex integrals
  if (ufc.form.has_vertex_integrals())
  {
    dolfin_error("LocalAssembler.cpp",
                 "assemble local problem",
                 "Local problem contains vertex integrals which are not yet "
                 "supported by LocalAssembler");
  }
}
//------------------------------------------------------------------------------
void
LocalAssembler::assemble_cell(Eigen::Matrix<double, Eigen::Dynamic,
                                            Eigen::Dynamic,
                                            Eigen::RowMajor>& A,
                              UFC& ufc,
                              const std::vector<double>& coordinate_dofs,
                              const ufc::cell& ufc_cell,
                              const Cell& cell,
                              const MeshFunction<std::size_t>* cell_domains)
{
  // Skip if there are no cell integrals
  if (!ufc.form.has_cell_integrals())
  {
    // Clear tensor here instead of in assemble() as a small speedup
    A.setZero();
    return;
  }

  // Extract default cell integral
  ufc::cell_integral* integral = ufc.default_cell_integral.get();

  // Get integral for sub domain (if any)
  if (cell_domains && !cell_domains->empty())
    integral = ufc.get_cell_integral((*cell_domains)[cell]);

  // Skip integral if zero
  if (!integral)
  {
    // Clear tensor here instead of in assemble() as a small speedup
    A.setZero();
    return;
  }

  // Update to current cell
  ufc.update(cell, coordinate_dofs, ufc_cell,
             integral->enabled_coefficients());

  // Tabulate cell tensor directly into A. This overwrites any
  // previous values
  integral->tabulate_tensor(A.data(), ufc.w(),
                            coordinate_dofs.data(),
                            ufc_cell.orientation);
}
//------------------------------------------------------------------------------
void
LocalAssembler::assemble_exterior_facet(Eigen::Matrix<double,
                                                      Eigen::Dynamic,
                                                      Eigen::Dynamic,
                                                      Eigen::RowMajor>& A,
                        UFC& ufc,
                        const std::vector<double>& coordinate_dofs,
                        const ufc::cell& ufc_cell,
                        const Cell& cell,
                        const Facet& facet,
                        const std::size_t local_facet,
                        const MeshFunction<std::size_t>* exterior_facet_domains)
{
  // Skip if there are no exterior facet integrals
  if (!ufc.form.has_exterior_facet_integrals())
    return;

  // Extract default exterior facet integral
  ufc::exterior_facet_integral* integral
    = ufc.default_exterior_facet_integral.get();

  // Get integral for sub domain (if any)
  if (exterior_facet_domains && !exterior_facet_domains->empty())
    integral = ufc.get_exterior_facet_integral((*exterior_facet_domains)[facet]);

  // Skip integral if zero
  if (!integral)
    return;

  // Update to current cell
  ufc.update(cell, coordinate_dofs, ufc_cell,
             integral->enabled_coefficients());

  // Tabulate exterior facet tensor. Here we cannot tabulate directly
  // into A since this will overwrite any previously assembled dx, ds
  // or dS forms
  integral->tabulate_tensor(ufc.A.data(),
                            ufc.w(),
                            coordinate_dofs.data(),
                            local_facet,
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
LocalAssembler::assemble_interior_facet(Eigen::Matrix<double, Eigen::Dynamic,
                                                      Eigen::Dynamic,
                                                      Eigen::RowMajor>& A,
                        UFC& ufc,
                        const std::vector<double>& coordinate_dofs,
                        const ufc::cell& ufc_cell,
                        const Cell& cell,
                        const Facet& facet,
                        const std::size_t local_facet,
                        const MeshFunction<std::size_t>* interior_facet_domains,
                        const MeshFunction<std::size_t>* cell_domains)
{
  // Skip if there are no interior facet integrals
  if (!ufc.form.has_interior_facet_integrals())
    return;

  // Extract default interior facet integral
  ufc::interior_facet_integral* integral
    = ufc.default_interior_facet_integral.get();

  // Get integral for sub domain (if any)
  if (interior_facet_domains && !interior_facet_domains->empty())
    integral = ufc.get_interior_facet_integral((*interior_facet_domains)[facet]);

  // Skip integral if zero
  if (!integral)
    return;

  // Extract mesh
  const Mesh& mesh = cell.mesh();
  const std::size_t D = mesh.topology().dim();

  // Get cells incident with facet (which is 0 and 1 here is
  // arbitrary)
  dolfin_assert(facet.num_entities(D) == 2);
  std::size_t cell_index_plus = facet.entities(D)[0];
  std::size_t cell_index_minus = facet.entities(D)[1];
  bool local_is_plus = cell_index_plus == cell.index();

  // The convention '+' = 0, '-' = 1 is from ffc
  const Cell cell0(mesh, cell_index_plus);
  const Cell cell1(mesh, cell_index_minus);

  // Is this facet on a domain boundary?
  if (cell_domains && !cell_domains->empty() &&
      (*cell_domains)[cell_index_plus] < (*cell_domains)[cell_index_minus])
  {
    std::swap(cell_index_plus, cell_index_minus);
  }

  // Get information about the adjacent cell
  const Cell& cell_adj = local_is_plus ? cell1 : cell0;
  std::vector<double> coordinate_dofs_adj;
  ufc::cell ufc_cell_adj;
  std::size_t local_facet_adj = cell_adj.index(facet);
  cell_adj.get_coordinate_dofs(coordinate_dofs_adj);
  cell_adj.get_cell_data(ufc_cell_adj);

  // Get information about plus and minus cells
  const std::vector<double>* coordinate_dofs0 = nullptr;
  const std::vector<double>* coordinate_dofs1 = nullptr;
  const ufc::cell* ufc_cell0 = nullptr;
  const ufc::cell* ufc_cell1 = nullptr;
  std::size_t local_facet0, local_facet1;
  if (local_is_plus)
  {
    coordinate_dofs0 = &coordinate_dofs;
    coordinate_dofs1 = &coordinate_dofs_adj;
    ufc_cell0 = &ufc_cell;
    ufc_cell1 = &ufc_cell_adj;
    local_facet0 = local_facet;
    local_facet1 = local_facet_adj;
  }
  else
  {
    coordinate_dofs1 = &coordinate_dofs;
    coordinate_dofs0 = &coordinate_dofs_adj;
    ufc_cell1 = &ufc_cell;
    ufc_cell0 = &ufc_cell_adj;
    local_facet1 = local_facet;
    local_facet0 = local_facet_adj;
  }

  // Update to current pair of cells and facets
  ufc.update(cell0, *coordinate_dofs0, *ufc_cell0,
             cell1, *coordinate_dofs1, *ufc_cell1,
             integral->enabled_coefficients());

  // Tabulate interior facet tensor on macro element
  integral->tabulate_tensor(ufc.macro_A.data(), ufc.macro_w(),
                            coordinate_dofs0->data(),
                            coordinate_dofs1->data(),
                            local_facet0, local_facet1,
                            ufc_cell0->orientation,
                            ufc_cell1->orientation);

  // Stuff upper left quadrant (corresponding to cell_plus) or lower
  // left quadrant (corresponding to cell_minus) into A depending on
  // which cell is the local cell
  const std::size_t M = A.rows();
  const std::size_t N = A.cols();
  const std::size_t offset_N = local_is_plus ? 0 : N;
  const std::size_t offset_M = local_is_plus ? 0 : M;
  if (N == 1)
  {
    for (std::size_t i = 0; i < M; i++)
      A(i, 0) += ufc.macro_A[i + offset_M];
  }
  else
  {
    for (std::size_t i = 0; i < M; i++)
      for (std::size_t j = 0; j < N; j++)
        A(i, j) += ufc.macro_A[2*N*(i + offset_M) + j + offset_N];
  }
}
//------------------------------------------------------------------------------
