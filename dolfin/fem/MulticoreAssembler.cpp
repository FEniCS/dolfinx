// Copyright (C) 2010 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Based on a prototype implementation by Didem Unat.
//
// First added:  2010-11-04
// Last changed: 2010-11-12

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <algorithm>
#include <sstream>

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
  // FIXME: in particular building the dof map. Note that we create
  // FIXME: a UFC object here and recreate one in each thread below.

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

  // Call multi-thread assembly
  assemble_threads(&A, &a, num_threads,
                   cell_domains, exterior_facet_domains, interior_facet_domains);

  // Finalize assembly of global tensor
  A.apply("add");
}
//-----------------------------------------------------------------------------
void MulticoreAssembler::assemble_threads(GenericTensor* A,
                                          const Form* a,
                                          uint num_threads,
                                          const MeshFunction<uint>* cell_domains,
                                          const MeshFunction<uint>* exterior_facet_domains,
                                          const MeshFunction<uint>* interior_facet_domains)
{
  info("Starting multi-core assembly with %d threads.", num_threads);

  // List of threads
  std::vector<boost::thread*> threads;
  std::vector<PStats> pstats(num_threads);

  // Start threads
  for (uint p = 0; p < num_threads; p++)
  {
    // Create thread
    boost::thread* thread =
      new boost::thread(boost::bind(assemble_thread,
                                    A, a, p, num_threads, &pstats[p],
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
    if (p > 0)
      pstats[0] += pstats[p];
  }

  // Display statistics
  cout << "Multi-core stats (total): " << pstats[0].str() << endl;
}
//-----------------------------------------------------------------------------
void MulticoreAssembler::assemble_thread(GenericTensor* A,
                                         const Form* a,
                                         uint thread_id,
                                         uint num_threads,
                                         PStats* pstats,
                                         const MeshFunction<uint>* cell_domains,
                                         const MeshFunction<uint>* exterior_facet_domains,
                                         const MeshFunction<uint>* interior_facet_domains)
{
  // FIXME: More stuff should be done here (in parallel) and not
  // FIXME: above in the serial assemble function.

  // Create data structure for local assembly data
  UFC ufc(*a);

  // Get local range
  const std::pair<uint, uint> range = MPI::local_range(thread_id, A->size(0), num_threads);

  info("Starting assembly in thread %d, range [%d, %d].",
       thread_id, range.first, range.second);

  // Assemble over cells
  assemble_cells(*A, *a, ufc, range, thread_id, *pstats, cell_domains, 0);

  // Assemble over exterior facets
  //assemble_exterior_facets(*A, *a, *ufc, range, thread_id, exterior_facet_domains, 0);

  // Assemble over interior facets
  //assemble_interior_facets(*A, *a, *ufc, range, thread_id, interior_facet_domains, 0);

  // Display statistics
  cout << "Multi-core stats (thread " << thread_id << "): " << pstats->str() << endl;
}
//-----------------------------------------------------------------------------
void MulticoreAssembler::assemble_cells(GenericTensor& A,
                                        const Form& a,
                                        UFC& ufc,
                                        const std::pair<uint, uint>& range,
                                        uint thread_id,
                                        PStats& pstats,
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
  ufc::cell_integral* integral = ufc.cell_integrals[0].get();

  // Assemble over cells
  //Progress p(AssemblerTools::progress_message(A.rank(), "cells"), mesh.num_cells());
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

    // Tabulate dofs for each dimension
    for (uint i = 0; i < ufc.form.rank(); i++)
      a.function_space(i)->dofmap().tabulate_dofs(ufc.dofs[i], ufc.cell, cell->index());

    // Check whether rows are in local range
    MulticoreAssembler::RangeCheck range_check = check_row_range(ufc, range, thread_id, pstats);

    // Skip if all rows are out-of-range
    if (range_check == none_in_range)
      continue;

    // Tabulate cell tensor
    integral->tabulate_tensor(ufc.A.get(), ufc.w, ufc.cell);

    // Add entries to global tensor
    if (values && ufc.form.rank() == 0)
    {
      // Store values cell-by-cell (currently only available for functionals)
      (*values)[cell->index()] = ufc.A[0];
    }
    else
    {
      // Extract relevant rows if not all rows are in range
      if (range_check == some_in_range)
        extract_row_range(ufc, range, thread_id);

      // Add to global tensor
      A.add(ufc.A.get(), ufc.local_dimensions.get(), ufc.dofs);
    }
    //p++;
  }
}
//-----------------------------------------------------------------------------
void MulticoreAssembler::assemble_exterior_facets(GenericTensor& A,
                                                  const Form& a,
                                                  UFC& ufc,
                                                  const std::pair<uint, uint>& range,
                                                  uint thread_id,
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
  ufc::exterior_facet_integral* integral = ufc.exterior_facet_integrals[0].get();

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
      //p++;
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

    //p++;
  }
}
//-----------------------------------------------------------------------------
void MulticoreAssembler::assemble_interior_facets(GenericTensor& A,
                                                  const Form& a,
                                                  UFC& ufc,
                                                  const std::pair<uint, uint>& range,
                                                  uint thread_id,
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
  ufc::interior_facet_integral* integral = ufc.interior_facet_integrals[0].get();

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
  //Progress p(AssemblerTools::progress_message(A.rank(), "interior facets"), mesh.num_facets());
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Only consider interior facets
    if (!facet->interior())
    {
      //p++;
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

    //p++;
  }
}
//-----------------------------------------------------------------------------
MulticoreAssembler::RangeCheck
MulticoreAssembler::check_row_range(const UFC& ufc,
                                    const std::pair<uint, uint>& range,
                                    uint thread_id,
                                    PStats& pstats)
{
  // Note: The thread_id argument is not really needed but is useful
  // for debugging.

  // Iterate over rows
  bool _all_in_range = true;
  bool _none_in_range = true;
  for (uint i = 0; i < ufc.local_dimensions[0]; i++)
  {
    const uint row = ufc.dofs[0][i];
    if (range.first <= row && row < range.second)
      _none_in_range = false;
    else
      _all_in_range = false;
  }

  // Check range
  RangeCheck range_check = some_in_range;
  if (_all_in_range)
  {
    range_check = all_in_range;
    pstats.num_all++;
  }
  else if (_none_in_range)
  {
    range_check = none_in_range;
    pstats.num_none++;
  }
  else
  {
    range_check = some_in_range;
    pstats.num_some++;
  }

  // Debugging
  /*
  std::stringstream s;
  s << "process " << thread_id << ": rows = ";
  for (uint i = 0; i < ufc.local_dimensions[0]; i++)
  {
    const uint row = ufc.dofs[0][i];
    s << " " << row;
  }
  s << " range = [" << range.first << ", " << range.second << "]";
  s << " range_check = " << range_check;
  cout << s.str() << endl;
  */

  return range_check;
}
//-----------------------------------------------------------------------------
void MulticoreAssembler::extract_row_range(UFC& ufc,
                                           const std::pair<uint, uint>& range,
                                           uint thread_id)
{
  // Note: The thread_id argument is not really needed but is useful
  // for debugging.

  // Compute stride
  uint stride = 1;
  for (uint i = 1; i < ufc.form.rank(); i++)
    stride *= ufc.local_dimensions[i];

  // Shrink list of row indices
  uint k = 0;
  double* block = ufc.A.get();
    for (uint i = 0; i < ufc.local_dimensions[0]; i++)
  {
    const uint row = ufc.dofs[0][i];
    if (range.first <= row && row < range.second)
    {
      ufc.dofs[0][k] = ufc.dofs[0][i];
      std::copy(block + i*stride, block + (i + 1)*stride, block + k*stride);
      k++;
    }
  }
  ufc.local_dimensions[0] = k;

  // Debugging
  /*
  std::stringstream s;
  s << "process " << thread_id << ": rows in range = ";
  for (uint i = 0; i < ufc.local_dimensions[0]; i++)
  {
    const uint row = ufc.dofs[0][i];
    s << " " << row;
  }
  cout << s.str() << endl;
  */
}
//-----------------------------------------------------------------------------
std::string MulticoreAssembler::PStats::str() const
{
  const uint total = num_all + num_some;
  const double fraction = static_cast<double>(num_some) / static_cast<double>(total);
  std::stringstream s;
  s << num_all  << " in range, "
    << num_some << " partially in range ("
    << (100.0*fraction) << "%)";
  return s.str();
}
//-----------------------------------------------------------------------------
