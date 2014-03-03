// Copyright (C) 2013 Anders Logg
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
// Last changed: 2014-03-03

#include <dolfin/function/MultiMeshFunctionSpace.h>

#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericLinearAlgebraFactory.h>
#include <dolfin/la/TensorLayout.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>

#include "SparsityPatternBuilder.h"
#include "UFC.h"
#include "Form.h"
#include "MultiMeshForm.h"
#include "MultiMeshDofMap.h"
#include "MultiMeshAssembler.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiMeshAssembler::MultiMeshAssembler()
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

  begin(PROGRESS, "Assembling tensor over MultiMesh function space.");

  // Initialize global tensor
  init_global_tensor(A, a);

  // Assemble over cells
  assemble_cells(A, a);

  // Finalize assembly of global tensor
  if (finalize_tensor)
    A.apply("add");

  end();
}
//-----------------------------------------------------------------------------
void MultiMeshAssembler::assemble_cells(GenericTensor& A, const MultiMeshForm& a)
{
  log(PROGRESS, "Assembling MultiMesh form over cells.");

  // Get form rank
  const std::size_t form_rank = a.rank();

  // Collect pointers to dof maps
  std::vector<const MultiMeshDofMap*> dofmaps;
  for (std::size_t i = 0; i < form_rank; i++)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  // Vector to hold dof map for a cell
  std::vector<const std::vector<dolfin::la_index>* > dofs(form_rank);

  // Iterate over parts
  ufc::cell ufc_cell;
  std::vector<double> vertex_coordinates;
  for (std::size_t part = 0; part < a.num_parts(); part++)
  {
    // Get form for current part
    const Form& a_part = *a.part(part);

    // Set current part for dofmaps
    for (std::size_t i = 0; i < form_rank; i++)
      dofmaps[i]->set_current_part(part);

    // Create data structure for local assembly data
    UFC ufc_part(a_part);

    // Extract mesh
    const Mesh& mesh_part = a_part.mesh();

    // Skip assembly if there are no cell integrals
    if (!ufc_part.form.has_cell_integrals())
      return;

    // Cell integral
    ufc::cell_integral* integral = ufc_part.default_cell_integral.get();

    // Iterate over cells
    for (CellIterator cell(mesh_part); !cell.end(); ++cell)
    {
      // Update to current cell
      cell->get_vertex_coordinates(vertex_coordinates);
      cell->get_cell_data(ufc_cell);
      ufc_part.update(*cell, vertex_coordinates, ufc_cell);

      // Get local-to-global dof maps for cell
      for (std::size_t i = 0; i < form_rank; ++i)
        dofs[i] = &(dofmaps[i]->cell_dofs(cell->index()));

      // Tabulate cell tensor
      integral->tabulate_tensor(ufc_part.A.data(), ufc_part.w(),
                                vertex_coordinates.data(),
                                ufc_cell.orientation);

      // Add entries to global tensor
      A.add(ufc_part.A.data(), dofs);
    }
  }
}
//-----------------------------------------------------------------------------
void MultiMeshAssembler::init_global_tensor(GenericTensor& A, const MultiMeshForm& a)
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
  std::vector<std::size_t> global_dimensions;
  std::vector<std::pair<std::size_t, std::size_t> > local_ranges;
  std::vector<std::size_t> block_sizes;
  for (std::size_t i = 0; i < a.rank(); i++)
  {
    std::shared_ptr<const MultiMeshFunctionSpace> V = a.function_space(i);
    dolfin_assert(V);

    global_dimensions.push_back(V->dim());
    local_ranges.push_back(std::make_pair(0, V->dim())); // FIXME: not parallel
  }

  // Set block size
  const std::size_t block_size = 1;

  // Initialise tensor layout
  tensor_layout->init(MPI_COMM_WORLD,
                      global_dimensions, block_size, local_ranges);

  // Build sparsity pattern if required
  if (tensor_layout->sparsity_pattern())
  {
    GenericSparsityPattern& pattern = *tensor_layout->sparsity_pattern();
    SparsityPatternBuilder::build_ccfem(pattern, a);
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
    GenericMatrix& _A = A.down_cast<GenericMatrix>();

    // Loop over rows and insert 0.0 on the diagonal
    const double block = 0.0;
    const std::pair<std::size_t, std::size_t> row_range = A.local_range(0);
    const std::size_t range = std::min(row_range.second, A.size(1));
    for (std::size_t i = row_range.first; i < range; i++)
    {
      dolfin::la_index _i = i;
      _A.set(&block, 1, &_i, 1, &_i);
    }
    A.apply("flush");
  }

  // Set tensor to zero
  A.zero();
}
//-----------------------------------------------------------------------------
