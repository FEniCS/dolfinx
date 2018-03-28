// Copyright (C) 2011 Marie E. Rognes
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "LocalAssembler.h"
#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/fem/UFC.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/MeshIterator.h>

using namespace dolfin;
using namespace dolfin::fem;

//------------------------------------------------------------------------------
void LocalAssembler::assemble(
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& A,
    UFC& ufc, const std::vector<double>& coordinate_dofs,
    const mesh::Cell& cell, const mesh::MeshFunction<std::size_t>* cell_domains,
    const mesh::MeshFunction<std::size_t>* exterior_facet_domains,
    const mesh::MeshFunction<std::size_t>* interior_facet_domains)
{
  // Assemble contributions from cell integral
  assemble_cell(A, ufc, coordinate_dofs, cell, cell_domains);

  // Assemble contributions from facet integrals
  if (ufc.dolfin_form.integrals().num_exterior_facet_integrals() > 0
      or ufc.dolfin_form.integrals().num_interior_facet_integrals() > 0)
  {
    unsigned int local_facet = 0;
    for (auto& facet : mesh::EntityRange<mesh::Facet>(cell))
    {
      const int Ncells = facet.num_entities(cell.dim());
      if (Ncells == 2)
      {
        assemble_interior_facet(A, ufc, coordinate_dofs, cell, facet,
                                local_facet, interior_facet_domains,
                                cell_domains);
      }
      else if (Ncells == 1)
      {
        assemble_exterior_facet(A, ufc, coordinate_dofs, cell, facet,
                                local_facet, exterior_facet_domains);
      }
      else
      {
        log::dolfin_error(
            "LocalAssembler.cpp", "assemble local problem",
            "Cell <-> facet connectivity not initialized, found "
            "facet with %d connected cells. Expected 1 or 2 cells",
            Ncells);
      }
      ++local_facet;
    }
  }

  // Check that there are no vertex integrals
  if (ufc.dolfin_form.integrals().num_vertex_integrals() > 0)
  {
    log::dolfin_error(
        "LocalAssembler.cpp", "assemble local problem",
        "Local problem contains vertex integrals which are not yet "
        "supported by LocalAssembler");
  }
}
//------------------------------------------------------------------------------
void LocalAssembler::assemble_cell(
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& A,
    UFC& ufc, const std::vector<double>& coordinate_dofs,
    const mesh::Cell& cell, const mesh::MeshFunction<std::size_t>* cell_domains)
{
  // Skip if there are no cell integrals
  if (ufc.dolfin_form.integrals().num_cell_integrals() == 0)
  {
    // Clear tensor here instead of in assemble() as a small speedup
    A.setZero();
    return;
  }

  // Extract default cell integral
  const ufc::cell_integral* integral
      = ufc.dolfin_form.integrals().cell_integral().get();

  // Get integral for sub domain (if any)
  // if (cell_domains && !cell_domains->empty())
  if (cell_domains)
  {
    integral = ufc.dolfin_form.integrals()
                   .cell_integral((*cell_domains)[cell])
                   .get();
  }

  // Skip integral if zero
  if (!integral)
  {
    // Clear tensor here instead of in assemble() as a small speedup
    A.setZero();
    return;
  }

  // Update to current cell
  ufc.update(cell, coordinate_dofs, integral->enabled_coefficients());

  // Tabulate cell tensor directly into A. This overwrites any
  // previous values
  integral->tabulate_tensor(A.data(), ufc.w(), coordinate_dofs.data(), 1);
}
//------------------------------------------------------------------------------
void LocalAssembler::assemble_exterior_facet(
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& A,
    UFC& ufc, const std::vector<double>& coordinate_dofs,
    const mesh::Cell& cell, const mesh::Facet& facet,
    const std::size_t local_facet,
    const mesh::MeshFunction<std::size_t>* exterior_facet_domains)
{
  // Skip if there are no exterior facet integrals
  if (ufc.dolfin_form.integrals().num_exterior_facet_integrals() == 0)
    return;

  // Extract default exterior facet integral
  const ufc::exterior_facet_integral* integral
      = ufc.dolfin_form.integrals().exterior_facet_integral().get();

  // Get integral for sub domain (if any)
  // if (exterior_facet_domains && !exterior_facet_domains->empty())
  if (exterior_facet_domains)
  {
    integral = ufc.dolfin_form.integrals()
                   .exterior_facet_integral((*exterior_facet_domains)[facet])
                   .get();
  }

  // Skip integral if zero
  if (!integral)
    return;

  // Update to current cell
  ufc.update(cell, coordinate_dofs, integral->enabled_coefficients());

  // Tabulate exterior facet tensor. Here we cannot tabulate directly
  // into A since this will overwrite any previously assembled dx, ds
  // or dS forms
  integral->tabulate_tensor(ufc.A.data(), ufc.w(), coordinate_dofs.data(),
                            local_facet, 1);

  // Stuff a_ufc.A into A
  const std::size_t M = A.rows();
  const std::size_t N = A.cols();
  for (std::size_t i = 0; i < M; i++)
    for (std::size_t j = 0; j < N; j++)
      A(i, j) += ufc.A[N * i + j];
}
//------------------------------------------------------------------------------
void LocalAssembler::assemble_interior_facet(
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& A,
    UFC& ufc, const std::vector<double>& coordinate_dofs,
    const mesh::Cell& cell, const mesh::Facet& facet,
    const std::size_t local_facet,
    const mesh::MeshFunction<std::size_t>* interior_facet_domains,
    const mesh::MeshFunction<std::size_t>* cell_domains)
{
  // Skip if there are no interior facet integrals
  if (ufc.dolfin_form.integrals().num_interior_facet_integrals() == 0)
    return;

  // Extract default interior facet integral
  const ufc::interior_facet_integral* integral
      = ufc.dolfin_form.integrals().interior_facet_integral().get();

  // Get integral for sub domain (if any)
  // if (interior_facet_domains && !interior_facet_domains->empty())
  if (interior_facet_domains)
  {
    integral = ufc.dolfin_form.integrals()
                   .interior_facet_integral((*interior_facet_domains)[facet])
                   .get();
  }

  // Skip integral if zero
  if (!integral)
    return;

  // Extract mesh
  const mesh::Mesh& mesh = cell.mesh();
  const std::size_t D = mesh.topology().dim();

  // Get cells incident with facet (which is 0 and 1 here is
  // arbitrary)
  dolfin_assert(facet.num_entities(D) == 2);
  std::int32_t cell_index_plus = facet.entities(D)[0];
  std::int32_t cell_index_minus = facet.entities(D)[1];
  bool local_is_plus = cell_index_plus == cell.index();

  // The convention '+' = 0, '-' = 1 is from ffc
  const mesh::Cell cell0(mesh, cell_index_plus);
  const mesh::Cell cell1(mesh, cell_index_minus);

  // Is this facet on a domain boundary?
  // if (cell_domains and !cell_domains->empty()
  //    and (*cell_domains)[cell_index_plus] <
  //    (*cell_domains)[cell_index_minus])
  if (cell_domains
      and (*cell_domains)[cell_index_plus] < (*cell_domains)[cell_index_minus])
  {
    std::swap(cell_index_plus, cell_index_minus);
  }

  // Get information about the adjacent cell
  const mesh::Cell& cell_adj = local_is_plus ? cell1 : cell0;
  std::vector<double> coordinate_dofs_adj;
  std::size_t local_facet_adj = cell_adj.index(facet);
  cell_adj.get_coordinate_dofs(coordinate_dofs_adj);

  // Get information about plus and minus cells
  const std::vector<double>* coordinate_dofs0 = nullptr;
  const std::vector<double>* coordinate_dofs1 = nullptr;
  std::size_t local_facet0, local_facet1;
  if (local_is_plus)
  {
    coordinate_dofs0 = &coordinate_dofs;
    coordinate_dofs1 = &coordinate_dofs_adj;
    local_facet0 = local_facet;
    local_facet1 = local_facet_adj;
  }
  else
  {
    coordinate_dofs1 = &coordinate_dofs;
    coordinate_dofs0 = &coordinate_dofs_adj;
    local_facet1 = local_facet;
    local_facet0 = local_facet_adj;
  }

  // Update to current pair of cells and facets
  ufc.update(cell0, *coordinate_dofs0, cell1, *coordinate_dofs1,
             integral->enabled_coefficients());

  // Tabulate interior facet tensor on macro element
  integral->tabulate_tensor(ufc.macro_A.data(), ufc.macro_w(),
                            coordinate_dofs0->data(), coordinate_dofs1->data(),
                            local_facet0, local_facet1, 1, 1);

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
        A(i, j) += ufc.macro_A[2 * N * (i + offset_M) + j + offset_N];
  }
}
//------------------------------------------------------------------------------
