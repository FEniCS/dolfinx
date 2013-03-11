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
// Modified by Anders Logg 2010-2013
//
// First added:  2010-11-10
// Last changed: 2013-02-26

#ifdef HAS_OPENMP

#include <algorithm>
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
#include "OpenMpAssembler.h"

using namespace dolfin;

//----------------------------------------------------------------------------
void OpenMpAssembler::assemble(GenericTensor& A, const Form& a)
{
  if (MPI::num_processes() > 1)
  {
    dolfin_error("OpenMPAssembler.cpp",
                 "perform multithreaded assembly using OpenMP assembler",
                 "The OpenMp assembler has not been tested in combination with MPI");
  }

   dolfin_assert(a.ufc_form());

  // All assembler functions above end up calling this function, which
  // in turn calls the assembler functions below to assemble over
  // cells, exterior and interior facets. Note the importance of
  // treating empty mesh functions as null pointers for the PyDOLFIN
  // interface.

  // Get cell domains
  const MeshFunction<std::size_t>* cell_domains = a.cell_domains().get();

  // Get exterior facet domains
  const MeshFunction<std::size_t>* exterior_facet_domains = a.exterior_facet_domains().get();

  // Get interior facet domains
  const MeshFunction<std::size_t>* interior_facet_domains = a.interior_facet_domains().get();

  // Check form
  AssemblerBase::check(a);

  // Create data structure for local assembly data
  UFC ufc(a);

  // Update off-process coefficients
  const std::vector<boost::shared_ptr<const GenericFunction> > coefficients = a.coefficients();
  for (std::size_t i = 0; i < coefficients.size(); ++i)
    coefficients[i]->update();

  // Initialize global tensor
  init_global_tensor(A, a);

  // FIXME: The below selections should be made robust

  if (a.ufc_form()->has_interior_facet_integrals())
    assemble_interior_facets(A, a, ufc, interior_facet_domains, 0);

  if (a.ufc_form()->has_exterior_facet_integrals())
    assemble_cells_and_exterior_facets(A, a, ufc, cell_domains,
                                       exterior_facet_domains, 0);
  else
    assemble_cells(A, a, ufc, cell_domains, 0);

  // Finalize assembly of global tensor
  if (finalize_tensor)
    A.apply("add");
}
//-----------------------------------------------------------------------------
void OpenMpAssembler::assemble_cells(GenericTensor& A, const Form& a,
                                     UFC& _ufc,
                                     const MeshFunction<std::size_t>* domains,
                                     std::vector<double>* values)
{
  // Skip assembly if there are no cell integrals
  if (!_ufc.form.has_cell_integrals())
    return;

  Timer timer("Assemble cells");

  // Set number of OpenMP threads (from parameter systems)
  const std::size_t num_threads = parameters["num_threads"];
  omp_set_num_threads(num_threads);

  // Extract mesh
  const Mesh& mesh = a.mesh();

  // FIXME: Check that UFC copy constructor is dealing with copying pointers correctly
  // Dummy UFC object since each thread needs to created its own UFC object
  UFC ufc(_ufc);

  // Form rank
  const std::size_t form_rank = ufc.form.rank();

  // Cell integral
  const ufc::cell_integral* integral = ufc.default_cell_integral.get();

  // Check whether integral is domain-dependent
  bool use_domains = domains && !domains->empty();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (std::size_t i = 0; i < form_rank; ++i)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  // Vector to hold dof map for a cell
  std::vector<const std::vector<dolfin::la_index>* > dofs(form_rank);

  // Color mesh
  std::vector<std::size_t> coloring_type = a.coloring(mesh.topology().dim());
  mesh.color(coloring_type);

  // Get coloring data
  std::map<const std::vector<std::size_t>,
           std::pair<std::vector<std::size_t>, std::vector<std::vector<std::size_t> > > >::const_iterator mesh_coloring;
  mesh_coloring = mesh.topology().coloring.find(coloring_type);
  if (mesh_coloring == mesh.topology().coloring.end())
  {
    dolfin_error("OpenMPAssembler.cpp",
                 "perform multithreaded assembly using OpenMP assembler",
                 "Requested mesh coloring has not been computed");
  }

  // Get coloring data
  const std::vector<std::vector<std::size_t> >& entities_of_color = mesh_coloring->second.second;

  // If assembling a scalar we need to ensure each threads assemble its own scalar
  std::vector<double> scalars(num_threads, 0.0);

  // Assemble over cells (loop over colours, then cells of same color)
  const std::size_t num_colors = entities_of_color.size();
  Progress p("Assembling cells (threaded)", num_colors);
  for (std::size_t color = 0; color < num_colors; ++color)
  {
    // Get the array of cell indices of current color
    const std::vector<std::size_t>& colored_cells = entities_of_color[color];

    // Number of cells of current color
    const int num_cells = colored_cells.size();

    // OpenMP test loop over cells of the same color
    #pragma omp parallel for schedule(guided, 20) firstprivate(ufc, dofs, integral)
    for (int cell_index = 0; cell_index < num_cells; ++cell_index)
    {
      // Cell index
      const std::size_t index = colored_cells[cell_index];

      // Create cell
      const Cell cell(mesh, index);

      // Get integral for sub domain (if any)
      if (use_domains)
        integral = ufc.get_cell_integral((*domains)[cell]);

      // Skip integral if zero
      if (!integral)
        continue;

      // Update to current cell
      ufc.update(cell);

      // Get local-to-global dof maps for cell
      for (std::size_t i = 0; i < form_rank; ++i)
        dofs[i] = &(dofmaps[i]->cell_dofs(index));

      // Tabulate cell tensor
      integral->tabulate_tensor(&ufc.A[0],
                                ufc.w(),
                                &ufc.cell.vertex_coordinates[0],
                                ufc.cell.orientation);

      // Add entries to global tensor
      if (values && form_rank == 0)
        (*values)[cell_index] = ufc.A[0];
      else if (form_rank == 0)
        scalars[omp_get_thread_num()] += ufc.A[0];
      else
        A.add(&ufc.A[0], dofs);
    }
    p++;
  }

  // If we assemble a scalar we need to sum the contributions from each thread
  if (form_rank == 0)
  {
    const double scalar_sum = std::accumulate(scalars.begin(), scalars.end(), 0.0);
    A.add(&scalar_sum, dofs);
  }
}
//-----------------------------------------------------------------------------
void OpenMpAssembler::assemble_cells_and_exterior_facets(GenericTensor& A,
          const Form& a, UFC& _ufc,
          const MeshFunction<std::size_t>* cell_domains,
          const MeshFunction<std::size_t>* exterior_facet_domains,
          std::vector<double>* values)
{
  Timer timer("Assemble cells and exterior facets");

  // Set number of OpenMP threads (from parameter systems)
  const int num_threads = parameters["num_threads"];
  omp_set_num_threads(num_threads);

  // Extract mesh
  const Mesh& mesh = a.mesh();

  // Compute facets and facet - cell connectivity if not already computed
  const std::size_t D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);
  dolfin_assert(mesh.ordered());

  // Get connectivity
  const MeshConnectivity& connectivity = mesh.topology()(D, D - 1);
  dolfin_assert(!connectivity.empty());

  // Dummy UFC object since each thread needs to created its own UFC object
  UFC ufc(_ufc);

  // Form rank
  const std::size_t form_rank = ufc.form.rank();

  // Cell and facet integrals
  ufc::cell_integral* cell_integral = ufc.default_cell_integral.get();
  ufc::exterior_facet_integral* facet_integral = ufc.default_exterior_facet_integral.get();

  // Check whether integrals are domain-dependent
  bool use_cell_domains = cell_domains && !cell_domains->empty();
  bool use_exterior_facet_domains = exterior_facet_domains && !exterior_facet_domains->empty();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (std::size_t i = 0; i < form_rank; ++i)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  // Vector to hold dof maps for a cell
  std::vector<const std::vector<dolfin::la_index>* > dofs(form_rank);

  // FIXME: Pass or determine coloring type
  // Define graph type
  std::vector<std::size_t> coloring_type = a.coloring(mesh.topology().dim());
  mesh.color(coloring_type);

  // Get coloring data
  std::map<const std::vector<std::size_t>,
           std::pair<std::vector<std::size_t>, std::vector<std::vector<std::size_t> > > >::const_iterator mesh_coloring;
  mesh_coloring = mesh.topology().coloring.find(coloring_type);
  if (mesh_coloring == mesh.topology().coloring.end())
  {
    dolfin_error("OpenMPAssembler.cpp",
                 "perform multithreaded assembly using OpenMP assembler",
                 "Requested mesh coloring has not been computed");
  }

  // Get coloring data
  const std::vector<std::vector<std::size_t> >& entities_of_color = mesh_coloring->second.second;

  // If assembling a scalar we need to ensure each threads assemble its own scalar
  std::vector<double> scalars(num_threads, 0.0);

  // Assemble over cells (loop over colors, then cells of same color)
  const std::size_t num_colors = entities_of_color.size();
  for (std::size_t color = 0; color < num_colors; ++color)
  {
    // Get the array of cell indices of current color
    const std::vector<std::size_t>& colored_cells = entities_of_color[color];

    // Number of cells of current color
    const int num_cell_in_color = colored_cells.size();

    // OpenMP test loop over cells of the same color
    Progress p(AssemblerBase::progress_message(A.rank(), "cells"), num_colors);
    #pragma omp parallel for schedule(guided, 20) firstprivate(ufc, dofs, cell_integral, facet_integral)
    for (int index = 0; index < num_cell_in_color; ++index)
    {
      // Cell index
      const std::size_t cell_index = colored_cells[index];

      // Create cell
      const Cell cell(mesh, cell_index);

      // Get integral for sub domain (if any)
      if (use_cell_domains)
        cell_integral = ufc.get_cell_integral((*cell_domains)[cell_index]);

      // Update to current cell
      ufc.update(cell);

      // Get local-to-global dof maps for cell
      for (std::size_t i = 0; i < form_rank; ++i)
        dofs[i] = &(dofmaps[i]->cell_dofs(cell_index));

      // Get number of entries in cell tensor
      std::size_t dim = 1;
      for (std::size_t i = 0; i < form_rank; ++i)
        dim *= dofs[i]->size();

      // Tabulate cell tensor if we have a cell_integral
      if (cell_integral)
        cell_integral->tabulate_tensor(&ufc.A[0],
                                       ufc.w(),
                                       &ufc.cell.vertex_coordinates[0],
                                       ufc.cell.orientation);
      else
        std::fill(ufc.A.begin(), ufc.A.end(), 0.0);

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
        const std::size_t local_facet = cell.index(*facet);

        // Get integral for sub domain (if any)
        if (use_exterior_facet_domains)
        {
          // Get global facet index
          const std::size_t facet_index = connectivity(cell_index)[local_facet];
          facet_integral = ufc.get_exterior_facet_integral((*exterior_facet_domains)[facet_index]);
        }

        // Skip integral if zero
        if (!facet_integral)
          continue;

        // FIXME: Do we really need an update version with the local
        //        facet index?
        // Update UFC object
        ufc.update(cell, local_facet);

        // Tabulate tensor
        facet_integral->tabulate_tensor(&ufc.A_facet[0],
                                        ufc.w(),
                                        &ufc.cell.vertex_coordinates[0],
                                        local_facet);

        // Add facet contribution
        for (std::size_t i = 0; i < dim; ++i)
          ufc.A[i] += ufc.A_facet[i];
      }

      // Add entries to global tensor
      if (values && form_rank == 0)
        (*values)[cell_index] = ufc.A[0];
      else if (form_rank == 0)
        scalars[omp_get_thread_num()] += ufc.A[0];
      else
        A.add(&ufc.A[0], dofs);
    }

    p++;
  }

  // If we assemble a scalar we need to sum the contributions from each thread
  if (form_rank == 0)
  {
    const double scalar_sum = std::accumulate(scalars.begin(), scalars.end(), 0.0);
    A.add(&scalar_sum, dofs);
  }
}
//-----------------------------------------------------------------------------
void OpenMpAssembler::assemble_interior_facets(GenericTensor& A, const Form& a,
                                         UFC& _ufc,
                                         const MeshFunction<std::size_t>* domains,
                                         std::vector<double>* values)
{
  warning("OpenMpAssembler::assemble_interior_facets is untested.");

  dolfin_assert(!values);

  // Skip assembly if there are no interior facet integrals
  if (!_ufc.form.has_interior_facet_integrals())
    return;

  Timer timer("Assemble interior facets");

  // Set number of OpenMP threads (from parameter systems)
  omp_set_num_threads(parameters["num_threads"]);

  // Get integral for sub domain (if any)

  bool use_domains = domains && !domains->empty();
  if (use_domains)
  {
    dolfin_error("OpenMPAssembler.cpp",
                 "perform multithreaded assembly using OpenMP assembler",
                 "Subdomains are not yet handled");
  }

  // Extract mesh
  const Mesh& mesh = a.mesh();

  // Color mesh
  std::vector<std::size_t> coloring_type = a.coloring(mesh.topology().dim() - 1);
  mesh.color(coloring_type);

  // Dummy UFC object since each thread needs to created its own UFC object
  UFC ufc(_ufc);

  // Form rank
  const std::size_t form_rank = ufc.form.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (std::size_t i = 0; i < form_rank; ++i)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  // Vector to hold dofs for cells
  std::vector<std::vector<dolfin::la_index> > macro_dofs(form_rank);

  // Interior facet integral
  const ufc::interior_facet_integral* integral = ufc.default_interior_facet_integral.get();

  // Compute facets and facet - cell connectivity if not already computed
  mesh.init(mesh.topology().dim() - 1);
  mesh.init(mesh.topology().dim() - 1, mesh.topology().dim());
  dolfin_assert(mesh.ordered());

  // Get interior facet directions (if any)
  boost::shared_ptr<MeshFunction<std::size_t> > facet_orientation = mesh.data().mesh_function("facet_orientation");
  if (facet_orientation && facet_orientation->dim() != mesh.topology().dim() - 1)
  {
    dolfin_error("OpenMPAssembler.cpp",
                 "perform multithreaded assembly using OpenMP assembler",
                 "Expecting facet orientation to be defined on facets (not dimension %d)",
                 facet_orientation->dim());
  }

  // Get coloring data
  std::map<const std::vector<std::size_t>,
           std::pair<std::vector<std::size_t>, std::vector<std::vector<std::size_t> > > >::const_iterator mesh_coloring;
  mesh_coloring = mesh.topology().coloring.find(coloring_type);

  // Check that requested coloring has been computed
  if (mesh_coloring == mesh.topology().coloring.end())
  {
    dolfin_error("OpenMPAssembler.cpp",
                 "perform multithreaded assembly using OpenMP assembler",
                 "Requested mesh coloring has not been computed");
  }

  // Get coloring data
  const std::vector<std::vector<std::size_t> >& entities_of_color = mesh_coloring->second.second;

  // Assemble over interior facets (loop over colours, then cells of same color)
  const std::size_t num_colors = entities_of_color.size();
  for (std::size_t color = 0; color < num_colors; ++color)
  {
    // Get the array of facet indices of current color
    const std::vector<std::size_t>& colored_facets = entities_of_color[color];

    // Number of facets of current color
    const int num_facets = colored_facets.size();

    // OpenMP test loop over cells of the same color
    Progress p(AssemblerBase::progress_message(A.rank(), "interior facets"), mesh.num_facets());
    #pragma omp parallel for schedule(guided, 20) firstprivate(ufc, macro_dofs, integral)
    for (int facet_index = 0; facet_index < num_facets; ++facet_index)
    {
      // Facet index
      const std::size_t index = colored_facets[facet_index];

      // Create cell
      const Facet facet(mesh, index);

      // Only consider interior facets
      if (facet.exterior())
      {
        p++;
        continue;
      }

      // Get integral for sub domain (if any)
      if (use_domains)
        integral = ufc.get_interior_facet_integral((*domains)[facet]);

      // Skip integral if zero
      if (!integral)
        continue;

      // Get cells incident with facet
      std::pair<const Cell, const Cell> cells = facet.adjacent_cells(facet_orientation.get());
      const Cell& cell0 = cells.first;
      const Cell& cell1 = cells.second;

      // Get local index of facet with respect to each cell
      const std::size_t local_facet0 = cell0.index(facet);
      const std::size_t local_facet1 = cell1.index(facet);

      // Update to current pair of cells
      ufc.update(cell0, local_facet0, cell1, local_facet1);

      // Tabulate dofs for each dimension on macro element
      for (std::size_t i = 0; i < form_rank; i++)
      {
        // Get dofs for each cell
        const std::vector<dolfin::la_index>& cell_dofs0 = dofmaps[i]->cell_dofs(cell0.index());
        const std::vector<dolfin::la_index>& cell_dofs1 = dofmaps[i]->cell_dofs(cell1.index());

        // Create space in macro dof vector
        macro_dofs[i].resize(cell_dofs0.size() + cell_dofs1.size());

        // Copy cell dofs into macro dof vector
        std::copy(cell_dofs0.begin(), cell_dofs0.end(), macro_dofs[i].begin());
        std::copy(cell_dofs1.begin(), cell_dofs1.end(),
                  macro_dofs[i].begin() + cell_dofs0.size());
      }

      // Tabulate exterior interior facet tensor on macro element
      integral->tabulate_tensor(&ufc.macro_A[0],
                                ufc.macro_w(),
                                &ufc.cell0.vertex_coordinates[0],
                                &ufc.cell1.vertex_coordinates[0],
                                local_facet0,
                                local_facet1);

      // Add entries to global tensor
      A.add(&ufc.macro_A[0], macro_dofs);

      p++;
    }
  }
}
//-----------------------------------------------------------------------------
#endif
