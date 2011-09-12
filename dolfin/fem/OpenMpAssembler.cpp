// Copyright (C) 2010-2011 Garth N. Wells
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
// Modified by Garth N. Wells 2007-2009
// Modified by Ola Skavhaug 2007-2009
// Modified by Kent-Andre Mardal 2008
// Modified by Anders Logg 2010-2011
//
// First added:  2010-11-10
// Last changed: 2011-08-10

#ifdef HAS_OPENMP

#include <map>
#include <utility>
#include <vector>
#include <omp.h>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/mesh/SubsetIterator.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/FunctionSpace.h>
#include "GenericDofMap.h"
#include "Form.h"
#include "UFC.h"
#include "FiniteElement.h"
#include "AssemblerTools.h"
#include "OpenMpAssembler.h"

using namespace dolfin;

//----------------------------------------------------------------------------
void OpenMpAssembler::assemble(GenericTensor& A,
                               const Form& a,
                               bool reset_sparsity,
                               bool add_values)
{
  assemble(A, a, 0, 0, 0, reset_sparsity, add_values);
}
//-----------------------------------------------------------------------------
void OpenMpAssembler::assemble(GenericTensor& A,
                               const Form& a,
                               const MeshFunction<uint>* cell_domains,
                               const MeshFunction<uint>* exterior_facet_domains,
                               const MeshFunction<uint>* interior_facet_domains,
                               bool reset_sparsity,
                               bool add_values)
{
  if (MPI::num_processes() > 1)
    error("OpenMpAssembler has not been tested in combination with MPI.");

   assert(a.ufc_form());

  // All assembler functions above end up calling this function, which
  // in turn calls the assembler functions below to assemble over
  // cells, exterior and interior facets. Note the importance of
  // treating empty mesh functions as null pointers for the PyDOLFIN
  // interface.

  // Check form
  AssemblerTools::check(a);

  // Create data structure for local assembly data
  UFC ufc(a);

  // Gather off-process coefficients
  const std::vector<boost::shared_ptr<const GenericFunction> > coefficients = a.coefficients();
  for (uint i = 0; i < coefficients.size(); ++i)
    coefficients[i]->gather();

  // Initialize global tensor
  AssemblerTools::init_global_tensor(A, a, reset_sparsity, add_values);

  // FIXME: The below selections should be made robust

  if (a.ufc_form()->num_interior_facet_domains() != 0)
    assemble_interior_facets(A, a, ufc, interior_facet_domains, 0);

  if (a.ufc_form()->num_exterior_facet_domains() != 0)
    assemble_cells_and_exterior_facets(A, a, ufc, exterior_facet_domains, 0);
  else
    assemble_cells(A, a, ufc, cell_domains, 0);

  // Finalize assembly of global tensor
  A.apply("add");
}
//-----------------------------------------------------------------------------
void OpenMpAssembler::assemble_cells(GenericTensor& A,
                                     const Form& a,
                                     UFC& _ufc,
                                     const MeshFunction<uint>* domains,
                                     std::vector<double>* values)
{
  // Skip assembly if there are no cell integrals
  if (_ufc.form.num_cell_domains() == 0)
    return;

  Timer timer("Assemble cells");

  // Set number of OpenMP threads (from parameter systems)
  omp_set_num_threads(parameters["num_threads"]);

  // Get integral for sub domain (if any)
  if (domains && domains->size() > 0)
    error("Sub-domains not yet handled by OpenMpAssembler.");

  // Extract mesh
  const Mesh& mesh = a.mesh();

  // FIXME: Check that UFC copy constructor is dealing with copying pointers correctly
  // Dummy UFC object since each thread needs to created its own UFC object
  UFC ufc(_ufc);

  // Form rank
  const uint form_rank = ufc.form.rank();

  // Cell integral
  const ufc::cell_integral* integral = ufc.cell_integrals[0].get();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (uint i = 0; i < form_rank; ++i)
    dofmaps.push_back(&a.function_space(i)->dofmap());

  // Vector to hold dof map for a cell
  std::vector<const std::vector<uint>* > dofs(form_rank);

  // Color mesh
  std::vector<uint> coloring_type = a.coloring(mesh.topology().dim());
  mesh.color(coloring_type);

  // Get coloring data
  std::map<const std::vector<uint>,
           std::pair<MeshFunction<uint>, std::vector<std::vector<uint> > > >::const_iterator mesh_coloring;
  mesh_coloring = mesh.parallel_data().coloring.find(coloring_type);
  if (mesh_coloring == mesh.parallel_data().coloring.end())
    error("Requested mesh coloring has not been computed. Cannot used multithreaded assembly.");

  // Get coloring data
  const std::vector<std::vector<uint> >& entities_of_color = mesh_coloring->second.second;

  // Assemble over cells (loop over colours, then cells of same color)
  const uint num_colors = entities_of_color.size();
  for (uint color = 0; color < num_colors; ++color)
  {
    // Get the array of cell indices of current color
    const std::vector<uint>& colored_cells = entities_of_color[color];

    // Number of cells of current color
    const uint num_cells = colored_cells.size();

    // OpenMP test loop over cells of the same color
    Progress p(AssemblerTools::progress_message(A.rank(), "cells"), num_colors);
    #pragma omp parallel for schedule(guided, 20) firstprivate(ufc, dofs, integral)
    for (uint cell_index = 0; cell_index < num_cells; ++cell_index)
    {
      // Cell index
      const uint index = colored_cells[cell_index];

      // Create cell
      const Cell cell(mesh, index);

      // Get integral for sub domain (if any)
      if (domains && domains->size() > 0)
      {
        const uint domain = (*domains)[cell];
        if (domain < ufc.form.num_cell_domains())
          integral = ufc.cell_integrals[domain].get();
        else
          continue;
      }

      // Skip integral if zero
      if (!integral)
        continue;

      // Update to current cell
      ufc.update(cell);

      // Get local-to-global dof maps for cell
      for (uint i = 0; i < form_rank; ++i)
        dofs[i] = &(dofmaps[i]->cell_dofs(index));

      // Tabulate cell tensor
      integral->tabulate_tensor(&ufc.A[0], ufc.w(), ufc.cell);

      // Add entries to global tensor
      if (values && form_rank == 0)
        (*values)[cell_index] = ufc.A[0];
      else
        A.add(&ufc.A[0], dofs);
    }
    p++;
  }
}
//-----------------------------------------------------------------------------
void OpenMpAssembler::assemble_cells_and_exterior_facets(GenericTensor& A,
                                         const Form& a,
                                         UFC& _ufc,
                                         const MeshFunction<uint>* domains,
                                         std::vector<double>* values)
{
  warning("OpenMpAssembler::assemble_cells_and_exterior_facets is untested.");

  // Get integral for sub domain (if any)
  if (domains && domains->size() > 0)
    error("Sub-domains not yet handled by OpenMpAssembler.");

  // Skip assembly if there are no exterior facet integrals
  if (_ufc.form.num_cell_domains() == 0 || _ufc.form.num_exterior_facet_domains() == 0)
    return;

  Timer timer("Assemble exterior facets");

  // Set number of OpenMP threads (from parameter systems)
  omp_set_num_threads(parameters["num_threads"]);

  // Extract mesh
  const Mesh& mesh = a.mesh();

  // Compute facets and facet - cell connectivity if not already computed
  const uint D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);
  assert(mesh.ordered());

  // Dummy UFC object since each thread needs to created its own UFC object
  UFC ufc(_ufc);

  // Form rank
  const uint form_rank = ufc.form.rank();

  // Cell and facet integrals
  const ufc::cell_integral* cell_integral = ufc.cell_integrals[0].get();
  const ufc::exterior_facet_integral* facet_integral = ufc.exterior_facet_integrals[0].get();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (uint i = 0; i < form_rank; ++i)
    dofmaps.push_back(&a.function_space(i)->dofmap());

  // Vector to hold dof maps for a cell
  std::vector<const std::vector<uint>* > dofs(form_rank);

  // FIXME: Pass or determine coloring type
  // Define graph type
  std::vector<uint> coloring_type = a.coloring(mesh.topology().dim());
  mesh.color(coloring_type);

  // Get coloring data
  std::map<const std::vector<uint>,
           std::pair<MeshFunction<uint>, std::vector<std::vector<uint> > > >::const_iterator mesh_coloring;
  mesh_coloring = mesh.parallel_data().coloring.find(coloring_type);
  if (mesh_coloring == mesh.parallel_data().coloring.end())
    error("Requested mesh coloring has not been computed. Cannot used multithreaded assembly.");

  // Get coloring data
  const std::vector<std::vector<uint> >& entities_of_color = mesh_coloring->second.second;

  // Assemble over cells (loop over colours, then cells of same color)
  const uint num_colors = entities_of_color.size();
  for (uint color = 0; color < num_colors; ++color)
  {
    // Get the array of cell indices of current color
    const std::vector<uint>& colored_cells = entities_of_color[color];

    // Number of cells of current color
    const uint num_cells = colored_cells.size();

    // OpenMP test loop over cells of the same color
    Progress p(AssemblerTools::progress_message(A.rank(), "cells"), num_colors);
    #pragma omp parallel for schedule(guided, 20) firstprivate(ufc, dofs, cell_integral, facet_integral)
    for (uint cell_index = 0; cell_index < num_cells; ++cell_index)
    {
      // Cell index
      const uint index = colored_cells[cell_index];

      // Create cell
      const Cell cell(mesh, index);

      // Get integral for sub domain (if any)
      if (domains && domains->size() > 0)
      {
        const uint domain = (*domains)[cell];
        if (domain < ufc.form.num_cell_domains())
          cell_integral = ufc.cell_integrals[domain].get();
        else
          continue;
      }

      // Skip integral if zero
      if (!cell_integral)
      {
        error("Need to fix this switch in OpenMP assembler");
        continue;
      }

      // Update to current cell
      ufc.update(cell);

      // Get local-to-global dof maps for cell
      for (uint i = 0; i < form_rank; ++i)
        dofs[i] = &(dofmaps[i]->cell_dofs(index));

      // Tabulate cell tensor
      cell_integral->tabulate_tensor(&ufc.A[0], ufc.w(), ufc.cell);

      // Get number of entries in cell tensor
      uint dim = 1;
      for (uint i = 0; i < form_rank; ++i)
        dim *= dofs[i]->size();

      // Assemble over external facet
      for (FacetIterator facet(cell); !facet.end(); ++facet)
      {
        // Only consider exterior facets
        if (!facet->exterior())
        {
          p++;
          continue;
        }

        // Get local facet index
        const uint local_facet = cell.index(*facet);
        //const ufc::exterior_facet_integral* facet_integral = ufc.exterior_facet_integrals[0].get();

        // FIXME: Do we really need an update version with the local facet index?
        // Update UFC object
        ufc.update(cell, local_facet);

        // Tabulate tensor
        facet_integral->tabulate_tensor(&ufc.A_facet[0], ufc.w(), ufc.cell, local_facet);

        // Add facet contribution
        for (uint i = 0; i < dim; ++i)
          ufc.A[i] += ufc.A_facet[i];
      }

      // Add entries to global tensor
      if (values && form_rank == 0)
        (*values)[cell_index] = ufc.A[0];
      else
        A.add(&ufc.A[0], dofs);
    }
    p++;
  }
}
//-----------------------------------------------------------------------------
void OpenMpAssembler::assemble_interior_facets(GenericTensor& A,
                                         const Form& a,
                                         UFC& _ufc,
                                         const MeshFunction<uint>* domains,
                                         std::vector<double>* values)
{
  warning("OpenMpAssembler::assemble_interior_facets is untested.");

  // Skip assembly if there are no interior facet integrals
  if (_ufc.form.num_interior_facet_domains() == 0)
    return;

  Timer timer("Assemble interior facets");

  // Set number of OpenMP threads (from parameter systems)
  omp_set_num_threads(parameters["num_threads"]);

  // Get integral for sub domain (if any)
  if (domains && domains->size() > 0)
    error("Sub-domains not yet handled by OpenMpAssembler.");

  // Extract mesh
  const Mesh& mesh = a.mesh();

  // Color mesh
  std::vector<uint> coloring_type = a.coloring(mesh.topology().dim() - 1);
  mesh.color(coloring_type);

  // Dummy UFC object since each thread needs to created its own UFC object
  UFC ufc(_ufc);

  // Form rank
  const uint form_rank = ufc.form.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (uint i = 0; i < form_rank; ++i)
    dofmaps.push_back(&a.function_space(i)->dofmap());

  // Vector to hold dofs for cells
  std::vector<std::vector<uint> > macro_dofs(form_rank);

  // Interior facet integral
  const ufc::interior_facet_integral* integral = ufc.interior_facet_integrals[0].get();

  // Compute facets and facet - cell connectivity if not already computed
  mesh.init(mesh.topology().dim() - 1);
  mesh.init(mesh.topology().dim() - 1, mesh.topology().dim());
  assert(mesh.ordered());

  // Get interior facet directions (if any)
  boost::shared_ptr<MeshFunction<unsigned int> > facet_orientation = mesh.data().mesh_function("facet_orientation");
  if (facet_orientation && facet_orientation->dim() != mesh.topology().dim() - 1)
  {
    error("Expecting facet orientation to be defined on facets (not dimension %d).",
          facet_orientation->dim());
  }

  // Get coloring data
  std::map<const std::vector<uint>,
           std::pair<MeshFunction<uint>, std::vector<std::vector<uint> > > >::const_iterator mesh_coloring;
  mesh_coloring = mesh.parallel_data().coloring.find(coloring_type);

  // Check that requested coloring has been computed
  if (mesh_coloring == mesh.parallel_data().coloring.end())
    error("Requested mesh coloring has not been computed. Cannot used multithreaded assembly.");

  // Get coloring data
  const std::vector<std::vector<uint> >& entities_of_color = mesh_coloring->second.second;

  // Assemble over interior facets (loop over colours, then cells of same color)
  const uint num_colors = entities_of_color.size();
  for (uint color = 0; color < num_colors; ++color)
  {
    // Get the array of facet indices of current color
    const std::vector<uint>& colored_facets = entities_of_color[color];

    // Number of facets of current color
    const uint num_facets = colored_facets.size();

    // OpenMP test loop over cells of the same color
    Progress p(AssemblerTools::progress_message(A.rank(), "interior facets"), mesh.num_facets());
    #pragma omp parallel for schedule(guided, 20) firstprivate(ufc, macro_dofs, integral)
    for (uint facet_index = 0; facet_index < num_facets; ++facet_index)
    {
      // Facet index
      const uint index = colored_facets[facet_index];

      // Create cell
      const Facet facet(mesh, index);

      // Only consider interior facets
      if (facet.exterior())
      {
        p++;
        continue;
      }

      // Get integral for sub domain (if any)
      if (domains && domains->size() > 0)
      {
        const uint domain = (*domains)[facet];
        if (domain < ufc.form.num_interior_facet_domains())
          integral = ufc.interior_facet_integrals[domain].get();
        else
          continue;
      }

      // Skip integral if zero
      if (!integral)
        continue;

      // Get cells incident with facet
      std::pair<const Cell, const Cell> cells = facet.adjacent_cells(facet_orientation.get());
      const Cell& cell0 = cells.first;
      const Cell& cell1 = cells.second;

      // Get local index of facet with respect to each cell
      const uint local_facet0 = cell0.index(facet);
      const uint local_facet1 = cell1.index(facet);

      // Update to current pair of cells
      ufc.update(cell0, local_facet0, cell1, local_facet1);

      // Tabulate dofs for each dimension on macro element
      for (uint i = 0; i < form_rank; i++)
      {
        // Get dofs for each cell
        const std::vector<uint>& cell_dofs0 = dofmaps[i]->cell_dofs(cell0.index());
        const std::vector<uint>& cell_dofs1 = dofmaps[i]->cell_dofs(cell1.index());

        // Create space in macro dof vector
        macro_dofs[i].resize(cell_dofs0.size() + cell_dofs1.size());

        // Copy cell dofs into macro dof vector
        std::copy(cell_dofs0.begin(), cell_dofs0.end(), macro_dofs[i].begin());
        std::copy(cell_dofs1.begin(), cell_dofs1.end(),
                  macro_dofs[i].begin() + cell_dofs0.size());
      }

      // Tabulate exterior interior facet tensor on macro element
      integral->tabulate_tensor(&ufc.macro_A[0], ufc.macro_w(),
                                ufc.cell0, ufc.cell1,
                                local_facet0, local_facet1);

      // Add entries to global tensor
      A.add(&ufc.macro_A[0], macro_dofs);

      p++;
    }
  }
}
//-----------------------------------------------------------------------------
#endif
