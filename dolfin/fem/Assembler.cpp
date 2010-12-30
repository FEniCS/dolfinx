// Copyright (C) 2007-2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2007-2009
// Modified by Ola Skavhaug, 2007-2009
// Modified by Kent-Andre Mardal, 2008
//
// First added:  2007-01-17
// Last changed: 2010-12-13

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
#include "AssemblerTools.h"
#include "Assembler.h"

using namespace dolfin;

//----------------------------------------------------------------------------
void Assembler::assemble(GenericTensor& A,
                         const Form& a,
                         bool reset_sparsity,
                         bool add_values)
{
  // Extract boundary indicators (if any)
  MeshFunction<uint>* exterior_facet_domains
    = a.mesh().data().mesh_function("exterior facet domains");

  // Assemble
  assemble(A, a, 0, exterior_facet_domains, 0, reset_sparsity, add_values);
}
//-----------------------------------------------------------------------------
void Assembler::assemble(GenericTensor& A,
                         const Form& a,
                         const SubDomain& sub_domain,
                         bool reset_sparsity,
                         bool add_values)
{
  // Extract mesh
  const Mesh& mesh = a.mesh();

  // Extract cell domains
  MeshFunction<uint>* cell_domains = 0;
  if (a.ufc_form().num_cell_integrals() > 0)
  {
    cell_domains = new MeshFunction<uint>(mesh, mesh.topology().dim(), 1);
    sub_domain.mark(*cell_domains, 0);
  }

  // Extract facet domains
  MeshFunction<uint>* facet_domains = 0;
  if (a.ufc_form().num_exterior_facet_integrals() > 0 ||
      a.ufc_form().num_interior_facet_integrals() > 0)
  {
    facet_domains = new MeshFunction<uint>(mesh, mesh.topology().dim() - 1, 1);
    sub_domain.mark(*facet_domains, 0);
  }

  // Assemble
  assemble(A, a, cell_domains, facet_domains, facet_domains, reset_sparsity, add_values);

  // Delete domains
  delete cell_domains;
  delete facet_domains;
}
//-----------------------------------------------------------------------------
void Assembler::assemble(GenericTensor& A,
                         const Form& a,
                         const MeshFunction<uint>* cell_domains,
                         const MeshFunction<uint>* exterior_facet_domains,
                         const MeshFunction<uint>* interior_facet_domains,
                         bool reset_sparsity,
                         bool add_values)
{
  // All assembler functions above end up calling this function, which
  // in turn calls the assembler functions below to assemble over
  // cells, exterior and interior facets. Note the importance of
  // treating empty mesh functions as null pointers for the PyDOLFIN
  // interface.

  // Check whether we should call the multi-core assembler
  #ifdef HAS_OPENMP
  const uint num_threads = parameters["num_threads"];
  if (num_threads > 0)
  {
    //MulticoreAssembler::assemble(A, a,
    //                             cell_domains,
    //                             exterior_facet_domains,
    //                             interior_facet_domains,
    //                             reset_sparsity, add_values, num_threads);

    OpenMpAssembler::assemble(A, a,
                              cell_domains,
                              exterior_facet_domains,
                              interior_facet_domains,
                              reset_sparsity, add_values);
    return;
  }
  #endif

  // Check form
  AssemblerTools::check(a);

  // Create data structure for local assembly data
  UFC ufc(a);

  // Gather off-process coefficients
  const std::vector<const GenericFunction*> coefficients = a.coefficients();
  for (uint i = 0; i < coefficients.size(); ++i)
    coefficients[i]->gather();

  // Initialize global tensor
  AssemblerTools::init_global_tensor(A, a, reset_sparsity, add_values);

  // Assemble over cells
  assemble_cells(A, a, ufc, cell_domains, 0);

  // Assemble over exterior facets
  assemble_exterior_facets(A, a, ufc, exterior_facet_domains, 0);

  // Assemble over interior facets
  assemble_interior_facets(A, a, ufc, interior_facet_domains, 0);

  // Finalize assembly of global tensor
  A.apply("add");
}
//-----------------------------------------------------------------------------
void Assembler::assemble_cells(GenericTensor& A,
                               const Form& a,
                               UFC& ufc,
                               const MeshFunction<uint>* domains,
                               std::vector<double>* values)
{
  // Skip assembly if there are no cell integrals
  if (ufc.form.num_cell_integrals() == 0)
    return;
  Timer timer("Assemble cells");

  // Extract mesh
  const Mesh& mesh = a.mesh();

  // Form rank
  const uint form_rank = ufc.form.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dof_maps;
  for (uint i = 0; i < form_rank; ++i)
    dof_maps.push_back(&a.function_space(i)->dofmap());

  // Vector to hold dof map for a cell
  std::vector<const std::vector<uint>* > dofs(form_rank);

  // Cell integral
  ufc::cell_integral* integral = ufc.cell_integrals[0].get();

  // Assemble over cells
  Progress p(AssemblerTools::progress_message(A.rank(), "cells"), mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get integral for sub domain (if any)
    if (domains && domains->size() > 0)
    {
      const uint domain = (*domains)[*cell];
      if (domain < ufc.form.num_cell_integrals())
        integral = ufc.cell_integrals[domain].get();
      else
        continue;
    }

    // Skip integral if zero
    if (!integral)
      continue;

    // Update to current cell
    ufc.update(*cell);

    // Get local-to-global dof maps for cell
    for (uint i = 0; i < form_rank; ++i)
      dofs[i] = &(dof_maps[i]->cell_dofs(cell->index()));

    // Tabulate cell tensor
    integral->tabulate_tensor(ufc.A.get(), ufc.w, ufc.cell);

    // Add entries to global tensor. Either store values cell-by-cell
    // (currently only available for functionals)
    if (values && ufc.form.rank() == 0)
      (*values)[cell->index()] = ufc.A[0];
    else
      A.add(ufc.A.get(), dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void Assembler::assemble_exterior_facets(GenericTensor& A,
                                         const Form& a,
                                         UFC& ufc,
                                         const MeshFunction<uint>* domains,
                                         std::vector<double>* values)
{
  // Skip assembly if there are no exterior facet integrals
  if (ufc.form.num_exterior_facet_integrals() == 0)
    return;
  Timer timer("Assemble exterior facets");

  // Extract mesh
  const Mesh& mesh = a.mesh();

  // Form rank
  const uint form_rank = ufc.form.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dof_maps;
  for (uint i = 0; i < form_rank; ++i)
    dof_maps.push_back(&a.function_space(i)->dofmap());

  // Vector to hold dof map for a cell
  std::vector<const std::vector<uint>* > dofs(form_rank);

  // Exterior facet integral
  const ufc::exterior_facet_integral* integral = ufc.exterior_facet_integrals[0].get();

  // Compute facets and facet - cell connectivity if not already computed
  const uint D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);
  assert(mesh.ordered());

  // Extract exterior (non shared) facets markers
  const MeshFunction<uint>* exterior_facets = mesh.data().mesh_function("exterior facets");

  // Assemble over exterior facets (the cells of the boundary)
  Progress p(AssemblerTools::progress_message(A.rank(), "exterior facets"), mesh.num_facets());
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Only consider exterior facets
    if (facet->num_entities(D) == 2 || (exterior_facets && !(*exterior_facets)[*facet]))
    {
      p++;
      continue;
    }

    // Get integral for sub domain (if any)
    if (domains && domains->size() > 0)
    {
      const uint domain = (*domains)[*facet];
      if (domain < ufc.form.num_exterior_facet_integrals())
        integral = ufc.exterior_facet_integrals[domain].get();
      else
        continue;
    }

    // Skip integral if zero
    if (!integral)
      continue;

    // Get mesh cell to which mesh facet belongs (pick first, there is only one)
    assert(facet->num_entities(mesh.topology().dim()) == 1);
    Cell mesh_cell(mesh, facet->entities(mesh.topology().dim())[0]);

    // Get local index of facet with respect to the cell
    const uint local_facet = mesh_cell.index(*facet);

    // Update to current cell
    ufc.update(mesh_cell, local_facet);

    // Get local-to-global dof maps for cell
    for (uint i = 0; i < form_rank; ++i)
      dofs[i] = &(dof_maps[i]->cell_dofs(mesh_cell.index()));

    // Tabulate exterior facet tensor
    integral->tabulate_tensor(ufc.A.get(), ufc.w, ufc.cell, local_facet);

    // Add entries to global tensor
    A.add(ufc.A.get(), dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void Assembler::assemble_interior_facets(GenericTensor& A,
                                         const Form& a,
                                         UFC& ufc,
                                         const MeshFunction<uint>* domains,
                                         std::vector<double>* values)
{
  // Skip assembly if there are no interior facet integrals
  if (ufc.form.num_interior_facet_integrals() == 0)
    return;

  Timer timer("Assemble interior facets");

  // Extract mesh and coefficients
  const Mesh& mesh = a.mesh();

  // Form rank
  const uint form_rank = ufc.form.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dof_maps;
  for (uint i = 0; i < form_rank; ++i)
    dof_maps.push_back(&a.function_space(i)->dofmap());

  // Vector to hold dofs for cells
  std::vector<std::vector<uint> > macro_dofs(form_rank);

  // Interior facet integral
  const ufc::interior_facet_integral* integral = ufc.interior_facet_integrals[0].get();

  // Compute facets and facet - cell connectivity if not already computed
  mesh.init(mesh.topology().dim() - 1);
  mesh.init(mesh.topology().dim() - 1, mesh.topology().dim());
  assert(mesh.ordered());

  // Get interior facet directions (if any)
  MeshFunction<uint>* facet_orientation = mesh.data().mesh_function("facet orientation");
  if (facet_orientation && facet_orientation->dim() != mesh.topology().dim() - 1)
  {
    error("Expecting facet orientation to be defined on facets (not dimension %d).",
          facet_orientation);
  }

  // Assemble over interior facets (the facets of the mesh)
  Progress p(AssemblerTools::progress_message(A.rank(), "interior facets"), mesh.num_facets());
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Only consider interior facets
    if (!facet->interior())
    {
      p++;
      continue;
    }

    // Get integral for sub domain (if any)
    if (domains && domains->size() > 0)
    {
      const uint domain = (*domains)[*facet];
      if (domain < ufc.form.num_interior_facet_integrals())
        integral = ufc.interior_facet_integrals[domain].get();
      else
        continue;
    }

    // Skip integral if zero
    if (!integral)
      continue;

    // Get cells incident with facet
    std::pair<const Cell, const Cell> cells = facet->adjacent_cells(facet_orientation);
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
      const std::vector<uint>& cell_dofs0 = dof_maps[i]->cell_dofs(cell0.index());
      const std::vector<uint>& cell_dofs1 = dof_maps[i]->cell_dofs(cell1.index());

      // Create space in macro dof vector
      macro_dofs[i].resize(cell_dofs0.size() + cell_dofs1.size());

      // Copy cell dofs into macro dof vector
      std::copy(cell_dofs0.begin(), cell_dofs0.end(), macro_dofs[i].begin());
      std::copy(cell_dofs1.begin(), cell_dofs1.end(), macro_dofs[i].begin() + cell_dofs0.size());
    }

    // Tabulate exterior interior facet tensor on macro element
    integral->tabulate_tensor(ufc.macro_A.get(), ufc.macro_w, ufc.cell0, ufc.cell1,
                              local_facet0, local_facet1);

    // Add entries to global tensor
    A.add(ufc.macro_A.get(), macro_dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
