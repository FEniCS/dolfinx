// Copyright (C) 2013-2014 Anders Logg
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
// First added:  2013-09-12
// Last changed: 2015-11-12

#include <dolfin/function/MultiMeshFunctionSpace.h>

#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericLinearAlgebraFactory.h>
#include <dolfin/la/TensorLayout.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MultiMesh.h>

#include "SparsityPatternBuilder.h"
#include "UFC.h"
#include "Form.h"
#include "MultiMeshForm.h"
#include "MultiMeshDofMap.h"
#include "MultiMeshAssembler.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiMeshAssembler::MultiMeshAssembler()
  : extend_cut_cell_integration(false)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MultiMeshAssembler::assemble(GenericTensor& A, const MultiMeshForm& a)
{
  // Developer note: This implementation does not yet handle
  // - subdomains
  // - interior facets
  // - exterior facets

  begin(PROGRESS, "Assembling tensor over multimesh function space.");

  // Initialize global tensor
  _init_global_tensor(A, a);

  // Assemble over uncut cells
  _assemble_uncut_cells(A, a);

  // Assemble over cut cells
  _assemble_cut_cells(A, a);

  // Assemble over interface
  _assemble_interface(A, a);

  // Assemble over overlap
  _assemble_overlap(A, a);

  // Finalize assembly of global tensor
  if (finalize_tensor)
    A.apply("add");

  // Lock any remaining inactive dofs
  if (A.rank() == 2)
    static_cast<GenericMatrix&>(A).ident_zeros();

  end();
}
//-----------------------------------------------------------------------------
void MultiMeshAssembler::_assemble_uncut_cells(GenericTensor& A,
                                               const MultiMeshForm& a)
{
  // Get form rank
  const std::size_t form_rank = a.rank();

  // Extract multimesh
  std::shared_ptr<const MultiMesh> multimesh = a.multimesh();

  // Collect pointers to dof maps
  std::vector<const MultiMeshDofMap*> dofmaps;
  for (std::size_t i = 0; i < form_rank; i++)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  // Vector to hold dof map for a cell
  std::vector<ArrayView<const dolfin::la_index>> dofs(form_rank);

  // Initialize variables that will be reused throughout assembly
  ufc::cell ufc_cell;
  std::vector<double> coordinate_dofs;

  // Iterate over parts
  for (std::size_t part = 0; part < a.num_parts(); part++)
  {
    log(PROGRESS, "Assembling multimesh form over uncut cells on part %d.", part);

    // Get form for current part
    const Form& a_part = *a.part(part);

    // Create data structure for local assembly data
    UFC ufc_part(a_part);

    // Extract mesh
    dolfin_assert(a_part.mesh());
    const Mesh& mesh_part = *(a_part.mesh());

    // FIXME: Handle subdomains

    // Get integral
    ufc::cell_integral* integral = ufc_part.default_cell_integral.get();

    // Skip if we don't have a cell integral
    if (!integral) continue;

    // Get uncut cells
    const std::vector<unsigned int>& uncut_cells = multimesh->uncut_cells(part);

    // Iterate over uncut cells
    for (auto it = uncut_cells.begin(); it != uncut_cells.end(); ++it)
    {
      // Create cell
      Cell cell(mesh_part, *it);

      // Update to current cell
      cell.get_cell_data(ufc_cell);
      cell.get_coordinate_dofs(coordinate_dofs);
      ufc_part.update(cell, coordinate_dofs, ufc_cell);

      // Get local-to-global dof maps for cell
      for (std::size_t i = 0; i < form_rank; ++i)
      {
        const auto dofmap = a.function_space(i)->dofmap()->part(part);
        dofs[i] = dofmap->cell_dofs(cell.index());
      }

      // Tabulate cell tensor
      integral->tabulate_tensor(ufc_part.A.data(),
                                ufc_part.w(),
                                coordinate_dofs.data(),
                                ufc_cell.orientation);

      // Add entries to global tensor
      A.add(ufc_part.A.data(), dofs);
    }
  }
}
//-----------------------------------------------------------------------------
void MultiMeshAssembler::_assemble_cut_cells(GenericTensor& A,
                                             const MultiMeshForm& a)
{
  // Get form rank
  const std::size_t form_rank = a.rank();

  // Extract multimesh
  std::shared_ptr<const MultiMesh> multimesh = a.multimesh();

  // Collect pointers to dof maps
  std::vector<const MultiMeshDofMap*> dofmaps;
  for (std::size_t i = 0; i < form_rank; i++)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  // Vector to hold dof map for a cell
  std::vector<ArrayView<const dolfin::la_index>> dofs(form_rank);

  // Initialize variables that will be reused throughout assembly
  ufc::cell ufc_cell;
  std::vector<double> coordinate_dofs;

  // Iterate over parts
  for (std::size_t part = 0; part < a.num_parts(); part++)
  {
    log(PROGRESS, "Assembling multimesh form over cut cells on part %d.", part);

    // Get form for current part
    const Form& a_part = *a.part(part);

    // Create data structure for local assembly data
    UFC ufc_part(a_part);

    // Extract mesh
    dolfin_assert(a_part.mesh());
    const Mesh& mesh_part = *(a_part.mesh());

    // FIXME: Handle subdomains

    // Get integral
    ufc::cutcell_integral* integral = ufc_part.default_cutcell_integral.get();

    // Skip if we don't have a cutcell integral
    if (!integral) continue;

    // Get cut cells and quadrature rules
    const std::vector<unsigned int>& cut_cells = multimesh->cut_cells(part);
    const auto& quadrature_rules = multimesh->quadrature_rule_cut_cells(part);

    // Iterate over cut cells
    for (auto it = cut_cells.begin(); it != cut_cells.end(); ++it)
    {
      // Create cell
      Cell cell(mesh_part, *it);

      // Update to current cell
      cell.get_cell_data(ufc_cell);
      cell.get_coordinate_dofs(coordinate_dofs);
      ufc_part.update(cell, coordinate_dofs, ufc_cell);

      // Get local-to-global dof maps for cell
      for (std::size_t i = 0; i < form_rank; ++i)
      {
        const auto dofmap = a.function_space(i)->dofmap()->part(part);
        dofs[i] = dofmap->cell_dofs(cell.index());
      }

      // Get quadrature rule for cut cell
      const auto& qr = quadrature_rules.at(*it);

      // Skip if there are no quadrature points
      std::size_t num_quadrature_points = qr.second.size();
      if (num_quadrature_points == 0)
        continue;

      // FIXME: Handle this inside the quadrature point generation,
      // FIXME: perhaps by storing three different sets of points,
      // FIXME: including cut cell, overlap and the whole cell.

      // Include only quadrature points with positive weight if
      // integration should be extended on cut cells
      std::pair<std::vector<double>, std::vector<double>> pr;
      if (extend_cut_cell_integration)
      {
        const std::size_t gdim = mesh_part.geometry().dim();
        for (std::size_t i = 0; i < num_quadrature_points; i++)
        {
          if (qr.second[i] > 0.0)
          {
            pr.second.push_back(qr.second[i]);
            for (std::size_t j = i*gdim; j < (i + 1)*gdim; j++)
              pr.first.push_back(qr.first[j]);
          }
        }
        num_quadrature_points = pr.second.size();
      }
      else
      {
        pr = qr;
      }

      // Tabulate cell tensor
      integral->tabulate_tensor(ufc_part.A.data(),
                                ufc_part.w(),
                                coordinate_dofs.data(),
                                num_quadrature_points,
                                pr.first.data(),
                                pr.second.data(),
                                ufc_cell.orientation);

      // Add entries to global tensor
      A.add(ufc_part.A.data(), dofs);
    }
  }
}
//-----------------------------------------------------------------------------
void MultiMeshAssembler::_assemble_interface(GenericTensor& A,
                                             const MultiMeshForm& a)
{
  // Extract multimesh
  std::shared_ptr<const MultiMesh> multimesh = a.multimesh();

  // Get form rank
  const std::size_t form_rank = a.rank();

  // Collect pointers to dof maps
  std::vector<const MultiMeshDofMap*> dofmaps;
  for (std::size_t i = 0; i < form_rank; i++)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  // Vector to hold dof map for a cell
  std::vector<const std::vector<dolfin::la_index>* > dofs(form_rank);

  // Initialize variables that will be reused throughout assembly
  ufc::cell ufc_cell[2];
  std::vector<double> coordinate_dofs[2];
  std::vector<double> macro_coordinate_dofs;

  // Vector to hold dofs for cells, and a vector holding pointers to same
  std::vector<ArrayView<const dolfin::la_index>> macro_dof_ptrs(form_rank);
  std::vector<std::vector<dolfin::la_index>> macro_dofs(form_rank);

  // Iterate over parts
  for (std::size_t part = 0; part < a.num_parts(); part++)
  {
    log(PROGRESS, "Assembling multimesh form over interface on part %d.",
        part);

    // Get form for current part
    const Form& a_part = *a.part(part);

    // Create data structure for local assembly data
    UFC ufc_part(a_part);

    // FIXME: Handle subdomains

    // Get integral
    ufc::interface_integral* integral = ufc_part.default_interface_integral.get();

    // Skip if we don't have an interface integral
    if (!integral) continue;

    // Get quadrature rules
    const auto& quadrature_rules = multimesh->quadrature_rule_interface(part);

    // Get collision map
    const auto& cmap = multimesh->collision_map_cut_cells(part);

    // Get facet normals
    const auto& facet_normals = multimesh->facet_normals(part);

    // Iterate over all cut cells in collision map
    for (auto it = cmap.begin(); it != cmap.end(); ++it)
    {
      // Get cut cell
      const unsigned int cut_cell_index = it->first;
      const Cell cut_cell(*multimesh->part(part), cut_cell_index);

      // Iterate over cutting cells
      const auto& cutting_cells = it->second;
      for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
      {
        // Get cutting part and cutting cell
        const std::size_t cutting_part = jt->first;
        const std::size_t cutting_cell_index = jt->second;
        const Cell cutting_cell(*multimesh->part(cutting_part),
                                cutting_cell_index);

        // Get quadrature rule for interface part defined by
        // intersection of the cut and cutting cells
        const std::size_t k = jt - cutting_cells.begin();
        dolfin_assert(k < quadrature_rules.at(cut_cell_index).size());
        const auto& qr = quadrature_rules.at(cut_cell_index)[k];

        // FIXME: There might be quite a few cases when we skip cutting
        // FIXME: cells because there are no quadrature points. Perhaps
        // FIXME: we can rewrite this inner loop to avoid unnecessary
        // FIXME: iterations.

        // Skip if there are no quadrature points
        const std::size_t num_quadrature_points = qr.second.size();
        if (num_quadrature_points == 0)
          continue;

        // Create aliases for cells to simplify notation
        const Cell& cell_0 = cut_cell;
        const Cell& cell_1 = cutting_cell;

        // Update to current pair of cells
        cell_0.get_cell_data(ufc_cell[0], 0);
        cell_1.get_cell_data(ufc_cell[1], 0);
        cell_0.get_coordinate_dofs(coordinate_dofs[0]);
        cell_1.get_coordinate_dofs(coordinate_dofs[1]);
        ufc_part.update(cell_0, coordinate_dofs[0], ufc_cell[0],
                        cell_1, coordinate_dofs[1], ufc_cell[1]);

        // Collect vertex coordinates
        macro_coordinate_dofs.resize(coordinate_dofs[0].size() +
                                     coordinate_dofs[0].size());
        std::copy(coordinate_dofs[0].begin(),
                  coordinate_dofs[0].end(),
                  macro_coordinate_dofs.begin());
        std::copy(coordinate_dofs[1].begin(),
                  coordinate_dofs[1].end(),
                  macro_coordinate_dofs.begin()
                  + coordinate_dofs[0].size());

        // Tabulate dofs for each dimension on macro element
        for (std::size_t i = 0; i < form_rank; i++)
        {
          // Get dofs for cut mesh
          const auto dofmap_0 = a.function_space(i)->dofmap()->part(part);
          const auto dofs_0 = dofmap_0->cell_dofs(cell_0.index());

          // Get dofs for cutting mesh
          const auto dofmap_1
            = a.function_space(i)->dofmap()->part(cutting_part);
          const auto dofs_1 = dofmap_1->cell_dofs(cell_1.index());

          // Create space in macro dof vector
          macro_dofs[i].resize(dofs_0.size() + dofs_1.size());

          // Copy cell dofs into macro dof vector
          std::copy(dofs_0.begin(), dofs_0.end(),
                    macro_dofs[i].begin());
          std::copy(dofs_1.begin(), dofs_1.end(),
                    macro_dofs[i].begin() + dofs_0.size());

          // Update array view
          macro_dof_ptrs[i]
            = ArrayView<const dolfin::la_index>(macro_dofs[i].size(),
                                                macro_dofs[i].data());
        }

        // Get facet normals
        const auto& n = facet_normals.at(cut_cell_index)[k];

        // FIXME: We would like to use this assertion (but it fails
        // for 2 meshes)
        dolfin_assert(n.size() == a_part.mesh()->geometry().dim()*num_quadrature_points);

        // FIXME: For now, use this assertion (which fails for 3 meshes)
        //dolfin_assert(n.size() > 0);

        // FIXME: Cell orientation not supported
        const int cell_orientation = ufc_cell[0].orientation;

        // Tabulate interface tensor on macro element
        integral->tabulate_tensor(ufc_part.macro_A.data(),
                                  ufc_part.macro_w(),
                                  macro_coordinate_dofs.data(),
                                  num_quadrature_points,
                                  qr.first.data(),
                                  qr.second.data(),
                                  n.data(),
                                  cell_orientation);

        // Add entries to global tensor
        A.add(ufc_part.macro_A.data(), macro_dof_ptrs);
      }
    }
  }
}
//-----------------------------------------------------------------------------
void MultiMeshAssembler::_assemble_overlap(GenericTensor& A,
                                           const MultiMeshForm& a)
{
  // FIXME: This function and assemble_interface are very similar.
  // FIXME: Refactor to improve code reuse.

  // Extract multimesh
  std::shared_ptr<const MultiMesh> multimesh = a.multimesh();

  // Get form rank
  const std::size_t form_rank = a.rank();

  // Collect pointers to dof maps
  std::vector<const MultiMeshDofMap*> dofmaps;
  for (std::size_t i = 0; i < form_rank; i++)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  // Vector to hold dof map for a cell
  std::vector<const std::vector<dolfin::la_index>* > dofs(form_rank);

  // Initialize variables that will be reused throughout assembly
  ufc::cell ufc_cell[2];
  std::vector<double> coordinate_dofs[2];
  std::vector<double> macro_coordinate_dofs;

  // Vector to hold dofs for cells, and a vector holding pointers to same
  std::vector<ArrayView<const dolfin::la_index>> macro_dof_ptrs(form_rank);
  std::vector<std::vector<dolfin::la_index>> macro_dofs(form_rank);

  // Iterate over parts
  for (std::size_t part = 0; part < a.num_parts(); part++)
  {
    log(PROGRESS, "Assembling multimesh form over overlap on part %d.", part);

    // Get form for current part
    const Form& a_part = *a.part(part);

    // Create data structure for local assembly data
    UFC ufc_part(a_part);

    // FIXME: Handle subdomains

    // Get integral
    ufc::overlap_integral* integral = ufc_part.default_overlap_integral.get();

    // Skip if we don't have an overlap integral
    if (!integral) continue;

    // Get quadrature rules
    const auto& quadrature_rules = multimesh->quadrature_rule_overlap(part);

    // Get collision map
    const auto& cmap = multimesh->collision_map_cut_cells(part);

    // Iterate over all cut cells in collision map
    for (auto it = cmap.begin(); it != cmap.end(); ++it)
    {
      // Get cut cell
      const unsigned int cut_cell_index = it->first;
      const Cell cut_cell(*multimesh->part(part), cut_cell_index);

      // Iterate over cutting cells
      const auto& cutting_cells = it->second;
      for (auto jt = cutting_cells.begin(); jt != cutting_cells.end(); jt++)
      {
        // Get cutting part and cutting cell
        const std::size_t cutting_part = jt->first;
        const std::size_t cutting_cell_index = jt->second;
        const Cell cutting_cell(*multimesh->part(cutting_part), cutting_cell_index);

        // Get quadrature rule for interface part defined by
        // intersection of the cut and cutting cells
        const std::size_t k = jt - cutting_cells.begin();
        dolfin_assert(k < quadrature_rules.at(cut_cell_index).size());
        const auto& qr = quadrature_rules.at(cut_cell_index)[k];

        // FIXME: There might be quite a few cases when we skip cutting
        // FIXME: cells because there are no quadrature points. Perhaps
        // FIXME: we can rewrite this inner loop to avoid unnecessary
        // FIXME: iterations.

        // Skip if there are no quadrature points
        const std::size_t num_quadrature_points = qr.second.size();
        if (num_quadrature_points == 0)
          continue;

        // Create aliases for cells to simplify notation
        const Cell& cell_0 = cut_cell;
        const Cell& cell_1 = cutting_cell;

        // Update to current pair of cells
        cell_0.get_cell_data(ufc_cell[0], 0);
        cell_1.get_cell_data(ufc_cell[1], 0);
        cell_0.get_coordinate_dofs(coordinate_dofs[0]);
        cell_1.get_coordinate_dofs(coordinate_dofs[1]);
        ufc_part.update(cell_0, coordinate_dofs[0], ufc_cell[0],
                        cell_1, coordinate_dofs[1], ufc_cell[1]);


        // Collect vertex coordinates
        macro_coordinate_dofs.resize(coordinate_dofs[0].size() +
                                     coordinate_dofs[0].size());
        std::copy(coordinate_dofs[0].begin(),
                  coordinate_dofs[0].end(),
                  macro_coordinate_dofs.begin());
        std::copy(coordinate_dofs[1].begin(),
                  coordinate_dofs[1].end(),
                  macro_coordinate_dofs.begin() + coordinate_dofs[0].size());

        // Tabulate dofs for each dimension on macro element
        for (std::size_t i = 0; i < form_rank; i++)
        {
          // Get dofs for cut mesh
          const auto dofmap_0 = a.function_space(i)->dofmap()->part(part);
          const auto dofs_0 = dofmap_0->cell_dofs(cell_0.index());

          // Get dofs for cutting mesh
          const auto dofmap_1 = a.function_space(i)->dofmap()->part(cutting_part);
          const auto dofs_1 = dofmap_1->cell_dofs(cell_1.index());

          // Create space in macro dof vector
          macro_dofs[i].resize(dofs_0.size() + dofs_1.size());

          // Copy cell dofs into macro dof vector
          std::copy(dofs_0.begin(), dofs_0.end(),
                    macro_dofs[i].begin());
          std::copy(dofs_1.begin(), dofs_1.end(),
                    macro_dofs[i].begin() + dofs_0.size());

          // Update array view
          macro_dof_ptrs[i]
            = ArrayView<const dolfin::la_index>(macro_dofs[i].size(),
                                                macro_dofs[i].data());
        }

        // FIXME: Cell orientation not supported
        const int cell_orientation = ufc_cell[0].orientation;

        // Tabulate overlap tensor on macro element
        integral->tabulate_tensor(ufc_part.macro_A.data(),
                                  ufc_part.macro_w(),
                                  macro_coordinate_dofs.data(),
                                  num_quadrature_points,
                                  qr.first.data(),
                                  qr.second.data(),
                                  cell_orientation);

        // Add entries to global tensor
        A.add(ufc_part.macro_A.data(), macro_dof_ptrs);
      }
    }
  }
}
//-----------------------------------------------------------------------------
void MultiMeshAssembler::_init_global_tensor(GenericTensor& A,
                                             const MultiMeshForm& a)
{
  log(PROGRESS, "Initializing global tensor.");

  // This function initializes the big system matrix corresponding to
  // all dofs (including inactive dofs) on all parts of the MultiMesh
  // function space.

  // Create layout for initializing tensor
  std::shared_ptr<TensorLayout> tensor_layout;
  tensor_layout = A.factory().create_layout(a.rank());
  dolfin_assert(tensor_layout);

  // Get dimensions
  std::vector<std::shared_ptr<const IndexMap>> index_maps;
  for (std::size_t i = 0; i < a.rank(); i++)
  {
    std::shared_ptr<const MultiMeshFunctionSpace> V = a.function_space(i);
    dolfin_assert(V);

    index_maps.push_back(std::shared_ptr<const IndexMap>
                         (new IndexMap(MPI_COMM_WORLD, V->dim(), 1)));
  }

  // Initialise tensor layout
  tensor_layout->init(MPI_COMM_WORLD, index_maps,
                      TensorLayout::Ghosts::UNGHOSTED);

  // Build sparsity pattern if required
  if (tensor_layout->sparsity_pattern())
  {
    SparsityPattern& pattern = *tensor_layout->sparsity_pattern();
    SparsityPatternBuilder::build_multimesh_sparsity_pattern(pattern, a);
  }

  // Initialize tensor
  A.init(*tensor_layout);

  // Insert zeros on the diagonal as diagonal entries may be prematurely
  // optimised away by the linear algebra backend when calling
  // GenericMatrix::apply, e.g. PETSc does this then errors when matrices
  // have no diagonal entry inserted.
  if (A.rank() == 2)
  {
    // Down cast to GenericMatrix
    GenericMatrix& _matA = A.down_cast<GenericMatrix>();

    // Loop over rows and insert 0.0 on the diagonal
    const double block = 0.0;
    const std::pair<std::size_t, std::size_t> row_range = A.local_range(0);
    const std::size_t range = std::min(row_range.second, A.size(1));
    for (std::size_t i = row_range.first; i < range; i++)
    {
      dolfin::la_index _i = i;
      _matA.set(&block, 1, &_i, 1, &_i);
    }
    A.apply("flush");
  }

  // Set tensor to zero
  A.zero();
}
//-----------------------------------------------------------------------------
