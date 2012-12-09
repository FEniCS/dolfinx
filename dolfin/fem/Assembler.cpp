// Copyright (C) 2007-2011 Anders Logg
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
// Modified by Joachim B Haga 2012
//
// First added:  2007-01-17
// Last changed: 2012-10-04

#include <boost/scoped_ptr.hpp>

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
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/FunctionSpace.h>
#include "GenericDofMap.h"
#include "Form.h"
#include "UFC.h"
#include "FiniteElement.h"
#include "OpenMpAssembler.h"
#include "AssemblerBase.h"
#include "Assembler.h"

#include <dolfin/la/GenericMatrix.h>

using namespace dolfin;

//----------------------------------------------------------------------------
void Assembler::assemble(GenericTensor& A, const Form& a)
{
  assemble(A, a, 0, 0, 0);
}
//-----------------------------------------------------------------------------
void Assembler::assemble(GenericTensor& A, const Form& a,
                         const SubDomain& sub_domain)
{
  dolfin_assert(a.ufc_form());

  // Extract mesh
  const Mesh& mesh = a.mesh();

  // Extract cell domains
  boost::scoped_ptr<MeshFunction<std::size_t> > cell_domains;
  if (a.ufc_form()->num_cell_domains() > 0)
  {
    cell_domains.reset(new MeshFunction<std::size_t>(mesh, mesh.topology().dim(), 1));
    sub_domain.mark(*cell_domains, 0);
  }

  // Extract facet domains
  boost::scoped_ptr<MeshFunction<std::size_t> > facet_domains;
  if (a.ufc_form()->num_exterior_facet_domains() > 0 ||
      a.ufc_form()->num_interior_facet_domains() > 0)
  {
    facet_domains.reset(new MeshFunction<std::size_t>(mesh, mesh.topology().dim() - 1, 1));
    sub_domain.mark(*facet_domains, 0);
  }

  // Assemble
  assemble(A, a, cell_domains.get(), facet_domains.get(), facet_domains.get());
}
//-----------------------------------------------------------------------------
void Assembler::assemble(GenericTensor& A,
                         const Form& a,
                         const MeshFunction<std::size_t>* cell_domains,
                         const MeshFunction<std::size_t>* exterior_facet_domains,
                         const MeshFunction<std::size_t>* interior_facet_domains)
{
  // All assembler functions above end up calling this function, which
  // in turn calls the assembler functions below to assemble over
  // cells, exterior and interior facets.
  //
  // Important notes:
  //
  // 1. Note the importance of treating empty mesh functions as null
  // pointers for the PyDOLFIN interface.
  //
  // 2. Note that subdomains given as input to this function override
  // subdomains attached to forms, which in turn override subdomains
  // stored as part of the mesh.

  // Get cell domains
  if (!cell_domains || cell_domains->empty())
  {
    cell_domains = a.cell_domains_shared_ptr().get();
    if (!cell_domains)
      cell_domains = a.mesh().domains().cell_domains(a.mesh()).get();
  }

  // Get exterior facet domains
  if (!exterior_facet_domains || exterior_facet_domains->empty())
  {
    exterior_facet_domains = a.exterior_facet_domains_shared_ptr().get();
    if (!exterior_facet_domains)
      exterior_facet_domains = a.mesh().domains().facet_domains(a.mesh()).get();
  }

  // Get interior facet domains
  if (!interior_facet_domains || interior_facet_domains->empty())
  {
    interior_facet_domains = a.interior_facet_domains_shared_ptr().get();
    if (!interior_facet_domains)
      interior_facet_domains = a.mesh().domains().facet_domains(a.mesh()).get();
  }

  // Check whether we should call the multi-core assembler
  #ifdef HAS_OPENMP
  const uint num_threads = parameters["num_threads"];
  if (num_threads > 0)
  {
    OpenMpAssembler assembler;
    assembler.reset_sparsity = reset_sparsity;
    assembler.add_values = add_values;
    assembler.finalize_tensor = finalize_tensor;
    assembler.keep_diagonal = keep_diagonal;
    assembler.assemble(A, a, cell_domains, exterior_facet_domains,
                       interior_facet_domains);
    return;
  }
  #endif

  // Check form
  AssemblerBase::check(a);

  // Create data structure for local assembly data
  UFC ufc(a);

  // Update off-process coefficients
  const std::vector<boost::shared_ptr<const GenericFunction> >
    coefficients = a.coefficients();
  for (uint i = 0; i < coefficients.size(); ++i)
    coefficients[i]->update();

  // Initialize global tensor
  const std::vector<std::pair<std::pair<std::size_t, std::size_t>, std::pair<std::size_t, std::size_t> > > periodic_master_slave_dofs;
  init_global_tensor(A, a, periodic_master_slave_dofs);

  // Assemble over cells
  assemble_cells(A, a, ufc, cell_domains, 0);

  // Assemble over exterior facets
  assemble_exterior_facets(A, a, ufc, exterior_facet_domains, 0);

  // Assemble over interior facets
  assemble_interior_facets(A, a, ufc, interior_facet_domains, 0);

  // Finalize assembly of global tensor
  if (finalize_tensor)
    A.apply("add");
}
//-----------------------------------------------------------------------------
void Assembler::assemble_cells(GenericTensor& A,
                               const Form& a,
                               UFC& ufc,
                               const MeshFunction<std::size_t>* domains,
                               std::vector<double>* values)
{
  // Skip assembly if there are no cell integrals
  if (ufc.form.num_cell_domains() == 0)
    return;

  // Set timer
  Timer timer("Assemble cells");

  // Extract mesh
  const Mesh& mesh = a.mesh();

  // Form rank
  const uint form_rank = ufc.form.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (uint i = 0; i < form_rank; ++i)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  // Vector to hold dof map for a cell
  std::vector<const std::vector<DolfinIndex>* > dofs(form_rank);

  // Cell integral
  dolfin_assert(!ufc.cell_integrals.empty());
  ufc::cell_integral* integral = ufc.cell_integrals[0].get();

  // Extract cell orientation
  boost::shared_ptr<MeshFunction<std::size_t> >
    cell_orientation = mesh.data().mesh_function("cell_orientation");

  // Assemble over cells
  Progress p(AssemblerBase::progress_message(A.rank(), "cells"), mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get integral for sub domain (if any)
    if (domains && !domains->empty())
    {
      const std::size_t domain = (*domains)[*cell];
      if (domain < ufc.form.num_cell_domains())
        integral = ufc.cell_integrals[domain].get();
      else
        continue;
    }

    // Skip if no integral on current domain
    if (!integral)
      continue;

    // Update to current cell
    if (cell_orientation)
      ufc.update(*cell, -1, (*cell_orientation)[cell->index()]);
    else
      ufc.update(*cell);

    // Get local-to-global dof maps for cell
    bool empty_dofmap = false;
    for (uint i = 0; i < form_rank; ++i)
    {
      dofs[i] = &(dofmaps[i]->cell_dofs(cell->index()));
      empty_dofmap = empty_dofmap || dofs[i]->size() == 0;
    }

    // Skip if at least one dofmap is empty
    if (empty_dofmap)
      continue;

    // Tabulate cell tensor
    integral->tabulate_tensor(&ufc.A[0], ufc.w(), ufc.cell);

    // Add entries to global tensor. Either store values cell-by-cell
    // (currently only available for functionals)
    if (values && ufc.form.rank() == 0)
      (*values)[cell->index()] = ufc.A[0];
    else
      add_to_global_tensor(A, ufc.A, dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void Assembler::assemble_exterior_facets(GenericTensor& A,
                                         const Form& a,
                                         UFC& ufc,
                                         const MeshFunction<std::size_t>* domains,
                                         std::vector<double>* values)
{
  // Skip assembly if there are no exterior facet integrals
  if (ufc.form.num_exterior_facet_domains() == 0)
    return;
  Timer timer("Assemble exterior facets");

  // Extract mesh
  const Mesh& mesh = a.mesh();

  // Form rank
  const uint form_rank = ufc.form.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (uint i = 0; i < form_rank; ++i)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  // Vector to hold dof map for a cell
  std::vector<const std::vector<DolfinIndex>* > dofs(form_rank);

  // Exterior facet integral
  dolfin_assert(!ufc.exterior_facet_integrals.empty());
  const ufc::exterior_facet_integral*
    integral = ufc.exterior_facet_integrals[0].get();

  // Compute facets and facet - cell connectivity if not already computed
  const uint D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);
  dolfin_assert(mesh.ordered());

  // Assemble over exterior facets (the cells of the boundary)
  Progress p(AssemblerBase::progress_message(A.rank(), "exterior facets"),
             mesh.num_facets());
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Only consider exterior facets
    if (!facet->exterior())
    {
      p++;
      continue;
    }

    // Get integral for sub domain (if any)
    if (domains && !domains->empty())
    {
      const std::size_t domain = (*domains)[*facet];
      if (domain < ufc.form.num_exterior_facet_domains())
        integral = ufc.exterior_facet_integrals[domain].get();
      else
        continue;
    }

    // Skip integral if zero
    if (!integral)
      continue;

    // Get mesh cell to which mesh facet belongs (pick first, there is only one)
    dolfin_assert(facet->num_entities(D) == 1);
    Cell mesh_cell(mesh, facet->entities(D)[0]);

    // Get local index of facet with respect to the cell
    const uint local_facet = mesh_cell.index(*facet);

    // Update to current cell
    ufc.update(mesh_cell, local_facet);

    // Get local-to-global dof maps for cell
    for (uint i = 0; i < form_rank; ++i)
      dofs[i] = &(dofmaps[i]->cell_dofs(mesh_cell.index()));

    // Tabulate exterior facet tensor
    integral->tabulate_tensor(&ufc.A[0], ufc.w(), ufc.cell, local_facet);

    // Add entries to global tensor
    add_to_global_tensor(A, ufc.A, dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void Assembler::assemble_interior_facets(GenericTensor& A,
                                         const Form& a,
                                         UFC& ufc,
                                         const MeshFunction<std::size_t>* domains,
                                         std::vector<double>* values)
{
  // Skip assembly if there are no interior facet integrals
  if (ufc.form.num_interior_facet_domains() == 0)
    return;

  not_working_in_parallel("Assembly over interior facets");

  Timer timer("Assemble interior facets");

  // Extract mesh and coefficients
  const Mesh& mesh = a.mesh();

  // Form rank
  const uint form_rank = ufc.form.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (uint i = 0; i < form_rank; ++i)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  // Vector to hold dofs for cells, and a vector holding pointers to same
  std::vector<std::vector<DolfinIndex> > macro_dofs(form_rank);
  std::vector<const std::vector<DolfinIndex>* > macro_dof_ptrs(form_rank);
  for (uint i = 0; i < form_rank; i++)
  {
    macro_dof_ptrs[i] = &macro_dofs[i];
  }

  // Interior facet integral
  dolfin_assert(!ufc.interior_facet_integrals.empty());
  const ufc::interior_facet_integral*
    integral = ufc.interior_facet_integrals[0].get();

  // Compute facets and facet - cell connectivity if not already computed
  const uint D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);
  dolfin_assert(mesh.ordered());

  // Get interior facet directions (if any)
  boost::shared_ptr<MeshFunction<std::size_t> >
    facet_orientation = mesh.data().mesh_function("facet_orientation");
  if (facet_orientation && facet_orientation->dim() != D - 1)
  {
    dolfin_error("Assembler.cpp",
                 "assemble form over interior facets",
                 "Expecting facet orientation to be defined on facets (not dimension %d)",
                 facet_orientation->dim());
  }

  // Assemble over interior facets (the facets of the mesh)
  Progress p(AssemblerBase::progress_message(A.rank(), "interior facets"),
             mesh.num_facets());
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Only consider interior facets
    if (facet->exterior())
    {
      p++;
      continue;
    }

    // Get integral for sub domain (if any)
    if (domains && !domains->empty())
    {
      const std::size_t domain = (*domains)[*facet];
      if (domain < ufc.form.num_interior_facet_domains())
        integral = ufc.interior_facet_integrals[domain].get();
      else
        continue;
    }

    // Skip integral if zero
    if (!integral)
      continue;

    // Get cells incident with facet
    std::pair<const Cell, const Cell>
      cells = facet->adjacent_cells(facet_orientation.get());
    const Cell& cell0 = cells.first;
    const Cell& cell1 = cells.second;

    // Get local index of facet with respect to each cell
    uint local_facet0 = cell0.index(*facet);
    uint local_facet1 = cell1.index(*facet);

    // Update to current pair of cells
    ufc.update(cell0, local_facet0, cell1, local_facet1);

    // Tabulate dofs for each dimension on macro element
    for (uint i = 0; i < form_rank; i++)
    {
      // Get dofs for each cell
      const std::vector<DolfinIndex>& cell_dofs0 = dofmaps[i]->cell_dofs(cell0.index());
      const std::vector<DolfinIndex>& cell_dofs1 = dofmaps[i]->cell_dofs(cell1.index());

      // Create space in macro dof vector
      macro_dofs[i].resize(cell_dofs0.size() + cell_dofs1.size());

      // Copy cell dofs into macro dof vector
      std::copy(cell_dofs0.begin(), cell_dofs0.end(),
                macro_dofs[i].begin());
      std::copy(cell_dofs1.begin(), cell_dofs1.end(),
                macro_dofs[i].begin() + cell_dofs0.size());
    }

    // Tabulate exterior interior facet tensor on macro element
    integral->tabulate_tensor(&ufc.macro_A[0], ufc.macro_w(), ufc.cell0, ufc.cell1,
                              local_facet0, local_facet1);

    // Add entries to global tensor
    add_to_global_tensor(A, ufc.macro_A, macro_dof_ptrs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void Assembler::add_to_global_tensor(GenericTensor& A,
                                     std::vector<double>& cell_tensor,
                                     std::vector<const std::vector<DolfinIndex>* >& dofs)
{
  A.add(&cell_tensor[0], dofs);
}
//-----------------------------------------------------------------------------
