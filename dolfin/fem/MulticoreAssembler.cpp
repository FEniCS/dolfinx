// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Based on a prototype implementation by Didem Unat.
//
// First added:  2010-11-04
// Last changed: 2010-11-08

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/main/MPI.h>
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
#include "MulticoreAssembler.h"
#include "AssemblerTools.h"
#include "MulticoreAssembler.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void MulticoreAssembler::assemble(GenericTensor& A,
                         const Form& a,
                         const MeshFunction<uint>* cell_domains,
                         const MeshFunction<uint>* exterior_facet_domains,
                         const MeshFunction<uint>* interior_facet_domains,
                         bool reset_sparsity,
                         bool add_values,
                         uint num_threads)
{
  // FIXME: Move more functionality to the assemble_thread below,
  // FIXME: in particular building the dof map

  // Check form
  AssemblerTools::check(a);

  // Create data structure for local assembly data
  UFC ufc(a);

  // Gather off-process coefficients
  const std::vector<const GenericFunction*> coefficients = a.coefficients();
  for (uint i = 0; i < coefficients.size(); ++i)
    coefficients[i]->gather();

  // Initialize global tensor
  AssemblerTools::init_global_tensor(A, a, ufc, reset_sparsity, add_values);

  // Call multi-thread assembly
  assemble_threads(&A, &a, &ufc, num_threads,
                   cell_domains, exterior_facet_domains, interior_facet_domains);

  // Finalize assembly of global tensor
  A.apply("add");
}
//-----------------------------------------------------------------------------
void MulticoreAssembler::assemble_threads(GenericTensor* A,
                                          const Form* a,
                                          UFC* ufc,
                                          uint num_threads,
                                          const MeshFunction<uint>* cell_domains,
                                          const MeshFunction<uint>* exterior_facet_domains,
                                          const MeshFunction<uint>* interior_facet_domains)
{
  info("Starting multi-core assembly with %d threads.", num_threads);

  // List of threads
  std::vector<boost::thread*> threads;

  // Start threads
  for (uint p = 0; p < num_threads; p++)
  {
    // Create thread
    boost::thread* thread =
      new boost::thread(boost::bind(assemble_thread,
                                    A, a, ufc, p, num_threads,
                                    cell_domains,
                                    exterior_facet_domains,
                                    interior_facet_domains));

    // Store thread
    threads.push_back(thread);
  }

  // Join threads
  for (uint p = 0; p < num_threads; p++)
  {
    threads[p]->join();
    delete threads[p];
  }
}
//-----------------------------------------------------------------------------
void MulticoreAssembler::assemble_thread(GenericTensor* A,
                                         const Form* a,
                                         UFC* ufc,
                                         uint thread_id,
                                         uint num_threads,
                                         const MeshFunction<uint>* cell_domains,
                                         const MeshFunction<uint>* exterior_facet_domains,
                                         const MeshFunction<uint>* interior_facet_domains)
{
  // FIXME: More stuff should be done here (in parallel) and not
  // FIXME: above in the serial assemble function.

  // Get local range
  const std::pair<uint, uint> range = MPI::local_range(thread_id, num_threads);

  info("Starting assembly in thread %d, range [%d, %d].",
       thread_id, range.first, range.second);

  // Assemble over cells
  assemble_cells(*A, *a, *ufc, range, cell_domains, 0);

  // Assemble over exterior facets
  //assemble_exterior_facets(*A, *a, *ufc, range, exterior_facet_domains, 0);

  // Assemble over interior facets
  //assemble_interior_facets(*A, *a, *ufc, range, interior_facet_domains, 0);
}
//-----------------------------------------------------------------------------
void MulticoreAssembler::assemble_cells(GenericTensor& A,
                                        const Form& a,
                                        UFC& ufc,
                                        const std::pair<uint, uint>& range,
                                        const MeshFunction<uint>* domains,
                                        std::vector<double>* values)
{
  // Skip assembly if there are no cell integrals
  if (ufc.form.num_cell_integrals() == 0)
    return;
  Timer timer("Assemble cells");

  // Extract mesh
  const Mesh& mesh = a.mesh();

  // Cell integral
  ufc::cell_integral* integral = ufc.cell_integrals[0];

  // Assemble over cells
  Progress p(AssemblerTools::progress_message(A.rank(), "cells"), mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get integral for sub domain (if any)
    if (domains && domains->size() > 0)
    {
      const uint domain = (*domains)[*cell];
      if (domain < ufc.form.num_cell_integrals())
        integral = ufc.cell_integrals[domain];
      else
        continue;
    }

    // Skip integral if zero
    if (!integral) continue;

    // Update to current cell
    ufc.update(*cell);

    // Tabulate dofs for each dimension
    for (uint i = 0; i < ufc.form.rank(); i++)
      a.function_space(i)->dofmap().tabulate_dofs(ufc.dofs[i], ufc.cell, cell->index());

    // Check whether rows are in local range
    bool all_in_range = true;
    bool none_in_range = true;
    for (uint i = 0; i < ufc.local_dimensions[0]; i++)
    {
      const uint row = ufc.dofs[0][i];
      if (range.first <= row && row < range.second)
        none_in_range = false;
      else
        all_in_range = false;
    }

    // FIXME: Temporary debuggin
    if (all_in_range)
      cout << "All in range" << endl;
    else if (none_in_range)
      cout << "None in range" << endl;
    else
      cout << "Some in range" << endl;

    // Skip if all rows are out-of-range
    //if (none_in_range)
    if (!all_in_range)
      continue;

    // Tabulate cell tensor
    integral->tabulate_tensor(ufc.A.get(), ufc.w, ufc.cell);

    // Add entries to global tensor
    if (values && ufc.form.rank() == 0)
    {
      // Either store values cell-by-cell (currently only available for functionals)
      (*values)[cell->index()] = ufc.A[0];
    }
    else
    {
      // Or add to the global tensor
      A.add(ufc.A.get(), ufc.local_dimensions.get(), ufc.dofs);
    }
    p++;
  }
}
//-----------------------------------------------------------------------------
void MulticoreAssembler::assemble_exterior_facets(GenericTensor& A,
                                                  const Form& a,
                                                  UFC& ufc,
                                                  const std::pair<uint, uint>& range,
                                                  const MeshFunction<uint>* domains,
                                                  std::vector<double>* values)
{
  dolfin_not_implemented();

  // Skip assembly if there are no exterior facet integrals
  if (ufc.form.num_exterior_facet_integrals() == 0)
    return;
  Timer timer("Assemble exterior facets");

  // Extract mesh
  const Mesh& mesh = a.mesh();

  // Exterior facet integral
  ufc::exterior_facet_integral* integral = ufc.exterior_facet_integrals[0];

  // Compute facets and facet - cell connectivity if not already computed
  const uint D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);
  assert(mesh.ordered());

  // Extract exterior (non shared) facets markers
  MeshFunction<uint>* exterior_facets = mesh.data().mesh_function("exterior facets");

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
        integral = ufc.exterior_facet_integrals[domain];
      else
        continue;
    }

    // Skip integral if zero
    if (!integral) continue;

    // Get mesh cell to which mesh facet belongs (pick first, there is only one)
    assert(facet->num_entities(mesh.topology().dim()) == 1);
    Cell mesh_cell(mesh, facet->entities(mesh.topology().dim())[0]);

    // Get local index of facet with respect to the cell
    const uint local_facet = mesh_cell.index(*facet);

    // Update to current cell
    ufc.update(mesh_cell, local_facet);

    // Tabulate dofs for each dimension
    for (uint i = 0; i < ufc.form.rank(); i++)
      a.function_space(i)->dofmap().tabulate_dofs(ufc.dofs[i], ufc.cell, mesh_cell.index());

    // Tabulate exterior facet tensor
    integral->tabulate_tensor(ufc.A.get(), ufc.w, ufc.cell, local_facet);

    // Add entries to global tensor
    A.add(ufc.A.get(), ufc.local_dimensions.get(), ufc.dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void MulticoreAssembler::assemble_interior_facets(GenericTensor& A,
                                                  const Form& a,
                                                  UFC& ufc,
                                                  const std::pair<uint, uint>& range,
                                                  const MeshFunction<uint>* domains,
                                                  std::vector<double>* values)
{
  dolfin_not_implemented();

  // Skip assembly if there are no interior facet integrals
  if (ufc.form.num_interior_facet_integrals() == 0)
    return;

  Timer timer("Assemble interior facets");

  // Extract mesh and coefficients
  const Mesh& mesh = a.mesh();

  // Interior facet integral
  ufc::interior_facet_integral* integral = ufc.interior_facet_integrals[0];

  // Compute facets and facet - cell connectivity if not already computed
  mesh.init(mesh.topology().dim() - 1);
  mesh.init(mesh.topology().dim() - 1, mesh.topology().dim());
  assert(mesh.ordered());

  // Get interior facet directions (if any)
  MeshFunction<uint>* facet_orientation = mesh.data().mesh_function("facet orientation");
  if (facet_orientation && facet_orientation->dim() != mesh.topology().dim() - 1)
    error("Expecting facet orientation to be defined on facets (not dimension %d).",
          facet_orientation);

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
        integral = ufc.interior_facet_integrals[domain];
      else
        continue;
    }

    // Skip integral if zero
    if (!integral) continue;

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
    for (uint i = 0; i < ufc.form.rank(); i++)
    {
      const uint offset = a.function_space(i)->dofmap().local_dimension(ufc.cell0);
      a.function_space(i)->dofmap().tabulate_dofs(ufc.macro_dofs[i],          ufc.cell0, cell0.index());
      a.function_space(i)->dofmap().tabulate_dofs(ufc.macro_dofs[i] + offset, ufc.cell1, cell1.index());
    }

    // Tabulate exterior interior facet tensor on macro element
    integral->tabulate_tensor(ufc.macro_A.get(), ufc.macro_w, ufc.cell0, ufc.cell1,
                              local_facet0, local_facet1);

    // Add entries to global tensor
    A.add(ufc.macro_A.get(), ufc.macro_local_dimensions.get(), ufc.macro_dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
