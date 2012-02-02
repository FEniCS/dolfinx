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
// First added: 2012-02-01 (modified from Assembler.cpp by jobh@simula.no)

#include <boost/scoped_ptr.hpp>

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/utils.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/la/GenericMatrix.h>
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
#include "AssemblerTools.h"
#include "SymmetricAssembler.h"

using namespace dolfin;

/// The private implementation class. It holds all relevant parameters for a
/// single assemble, the implementation, and some scratch variables. Its
/// lifetime is never longer than the assemble itself, so it's safe to keep
/// references to parameters.
class SymmetricAssembler::PImpl
{
public:
  // User-provided parameters
  GenericMatrix &A, &A_nonsymm;
  const Form &a;
  const std::vector<const DirichletBC*> &row_bcs, &col_bcs;
  const MeshFunction<uint> *cell_domains, *exterior_facet_domains, *interior_facet_domains;
  bool reset_sparsity, add_values, finalize_tensor;

  PImpl(GenericMatrix &_A, GenericMatrix &_A_nonsymm,
        const Form &_a,
        const std::vector<const DirichletBC*> &_row_bcs,
        const std::vector<const DirichletBC*> &_col_bcs,
        const MeshFunction<uint> *_cell_domains,
        const MeshFunction<uint> *_exterior_facet_domains,
        const MeshFunction<uint> *_interior_facet_domains,
        bool _reset_sparsity, bool _add_values, bool _finalize_tensor)
    : A(_A), A_nonsymm(_A_nonsymm), a(_a),
      row_bcs(_row_bcs), col_bcs(_col_bcs),
      cell_domains(_cell_domains),
      exterior_facet_domains(_exterior_facet_domains),
      interior_facet_domains(_interior_facet_domains),
      reset_sparsity(_reset_sparsity),
      add_values(_add_values),
      finalize_tensor(_finalize_tensor),
      ufc(_a), ufc_nonsymm(_a), mesh(_a.mesh())
  {
    init();
  }

  void assemble();

private:
  void init();
  void assemble_cells();
  void assemble_exterior_facets();
  void assemble_interior_facets();

  void apply_local_bc(std::vector<double> &elm_A, std::vector<double> &elm_A_nonsymm,
                      const std::vector<const std::vector<uint>*> &dofs);

  // These are derived from the variables above:
  const Mesh &mesh;     // = Mesh(a)
  UFC ufc;              // = UFC(a)
  UFC ufc_nonsymm;      // = UFC(a), used for scratch local tensors
  bool matching_bcs;    // true if row_bcs==col_bcs
  DirichletBC::Map row_bc_values; // derived from row_bcs
  DirichletBC::Map col_bc_values; // derived from col_bcs, but empty if matching_bcs

  // Used to ensure that each diagonal entry is set only once. An alternative
  // is to leave the diagonal unset, and use DirichletBC::apply afterwards to
  // set the diagonal to 1.0.
  std::set<uint> already_set_diagonals;

  // Scratch variables
  std::vector<bool> local_row_is_bc;
};
//-----------------------------------------------------------------------------
void SymmetricAssembler::assemble(GenericMatrix& A,
                                  GenericMatrix& A_nonsymm,
                                  const Form& a,
                                  const std::vector<const DirichletBC*> &row_bcs,
                                  const std::vector<const DirichletBC*> &col_bcs,
                                  const MeshFunction<uint>* cell_domains,
                                  const MeshFunction<uint>* exterior_facet_domains,
                                  const MeshFunction<uint>* interior_facet_domains,
                                  bool reset_sparsity,
                                  bool add_values,
                                  bool finalize_tensor)
{
  PImpl pImpl(A, A_nonsymm, a, row_bcs, col_bcs,
            cell_domains, exterior_facet_domains, interior_facet_domains,
            reset_sparsity, add_values, finalize_tensor);
  pImpl.assemble();
}
//-----------------------------------------------------------------------------
void SymmetricAssembler::PImpl::init()
{
  // If the bcs match (which is the usual case), we are assembling a normal
  // square matrix which contains the diagonal (and the dofmaps should match,
  // too).
  matching_bcs = (row_bcs == col_bcs);

  // Get Dirichlet dofs rows and values for local mesh
  for (uint i = 0; i < row_bcs.size(); ++i)
  {
    // Methods other than 'pointwise' are not robust in parallel since a vertex
    // can have a bc applied, but the partition might not have a facet on the
    // boundary.
    if (row_bcs[i]->method() != "pointwise"
        && dolfin::MPI::num_processes() > 1
        && dolfin::MPI::process_number() == 0)
      warning("Dirichlet boundary condition method '%s' is not robust in parallel"
              " with symmetric assembly.", row_bcs[i]->method().c_str());

    row_bcs[i]->get_boundary_values(row_bc_values);
  }
  if (!matching_bcs)
  {
    // Get Dirichlet dofs columns and values for local mesh
    for (uint i = 0; i < col_bcs.size(); ++i)
    {
      if (col_bcs[i]->method() != "pointwise"
          && dolfin::MPI::num_processes() > 1
          && dolfin::MPI::process_number() == 0)
        warning("Dirichlet boundary condition method '%s' is not robust in parallel"
                " with symmetric assembly.", col_bcs[i]->method().c_str());

      col_bcs[i]->get_boundary_values(col_bc_values);
    }
  }

  dolfin_assert(a.rank() == 2);
}
//-----------------------------------------------------------------------------
void SymmetricAssembler::PImpl::assemble()
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
  if (!cell_domains || cell_domains->size() == 0)
  {
    cell_domains = a.cell_domains_shared_ptr().get();
    if (!cell_domains)
      cell_domains = a.mesh().domains().cell_domains(a.mesh()).get();
  }

  // Get exterior facet domains
  if (!exterior_facet_domains || exterior_facet_domains->size() == 0)
  {
    exterior_facet_domains = a.exterior_facet_domains_shared_ptr().get();
    if (!exterior_facet_domains)
      exterior_facet_domains = a.mesh().domains().facet_domains(a.mesh()).get();
  }

  // Get interior facet domains
  if (!interior_facet_domains || interior_facet_domains->size() == 0)
  {
    interior_facet_domains = a.interior_facet_domains_shared_ptr().get();
    if (!interior_facet_domains)
      interior_facet_domains = a.mesh().domains().facet_domains(a.mesh()).get();
  }

  // Check form
  AssemblerTools::check(a);

  // Gather off-process coefficients
  const std::vector<boost::shared_ptr<const GenericFunction> >
    coefficients = a.coefficients();
  for (uint i = 0; i < coefficients.size(); ++i)
    coefficients[i]->gather();

  // Initialize global tensors
  AssemblerTools::init_global_tensor(A, a, 0, reset_sparsity, add_values);
  AssemblerTools::init_global_tensor(A_nonsymm, a, 0, reset_sparsity, add_values);

  // Assemble over cells
  assemble_cells();

  // Assemble over exterior facets
  assemble_exterior_facets();

  // Assemble over interior facets
  assemble_interior_facets();

  // Finalize assembly of global tensor
  if (finalize_tensor)
  {
    A.apply("add");
    A_nonsymm.apply("add");
  }
}
//-----------------------------------------------------------------------------
void SymmetricAssembler::PImpl::assemble_cells()
{
  // Skip assembly if there are no cell integrals
  if (ufc.form.num_cell_domains() == 0)
    return;

  // Set timer
  Timer timer("Assemble cells");

  // Form rank
  const uint form_rank = ufc.form.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> dofmaps;
  for (uint i = 0; i < form_rank; ++i)
    dofmaps.push_back(a.function_space(i)->dofmap().get());

  // Vector to hold dof map for a cell
  std::vector<const std::vector<uint>* > dofs(form_rank);

  // Cell integral
  dolfin_assert(ufc.cell_integrals.size() > 0);
  ufc::cell_integral* integral = ufc.cell_integrals[0].get();

  // Assemble over cells
  Progress p(AssemblerTools::progress_message(A.rank(), "cells"), mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Get integral for sub domain (if any)
    if (cell_domains && cell_domains->size() > 0)
    {
      const uint domain = (*cell_domains)[*cell];
      if (domain < ufc.form.num_cell_domains())
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
      dofs[i] = &(dofmaps[i]->cell_dofs(cell->index()));

    // Tabulate cell tensor
    integral->tabulate_tensor(&ufc.A[0], ufc.w(), ufc.cell);

    // Apply boundary conditions
    apply_local_bc(ufc.A, ufc_nonsymm.A, dofs);

    // Add entries to global tensor.
    A.add(&ufc.A[0], dofs);
    A_nonsymm.add(&ufc_nonsymm.A[0], dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void SymmetricAssembler::PImpl::assemble_exterior_facets()
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
  std::vector<const std::vector<uint>* > dofs(form_rank);

  // Exterior facet integral
  dolfin_assert(ufc.exterior_facet_integrals.size() > 0);
  const ufc::exterior_facet_integral*
    integral = ufc.exterior_facet_integrals[0].get();

  // Compute facets and facet - cell connectivity if not already computed
  const uint D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);
  dolfin_assert(mesh.ordered());

  // Assemble over exterior facets (the cells of the boundary)
  Progress p(AssemblerTools::progress_message(A.rank(), "exterior facets"),
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
    if (exterior_facet_domains && exterior_facet_domains->size() > 0)
    {
      const uint domain = (*exterior_facet_domains)[*facet];
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

    // Apply boundary conditions
    apply_local_bc(ufc.A, ufc_nonsymm.A, dofs);

    // Add entries to global tensor
    A.add(&ufc.A[0], dofs);
    A_nonsymm.add(&ufc_nonsymm.A[0], dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void SymmetricAssembler::PImpl::assemble_interior_facets()
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

  // Vector to hold dofs for cells
  std::vector<std::vector<uint> > macro_dofs(form_rank);
  std::vector<const std::vector<uint>*> macro_dof_ptrs(form_rank);
  for (uint i = 0; i < form_rank; i++)
    macro_dof_ptrs[i] = &macro_dofs[i];

  // Interior facet integral
  dolfin_assert(ufc.interior_facet_integrals.size() > 0);
  const ufc::interior_facet_integral*
    integral = ufc.interior_facet_integrals[0].get();

  // Compute facets and facet - cell connectivity if not already computed
  const uint D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);
  dolfin_assert(mesh.ordered());

  // Get interior facet directions (if any)
  boost::shared_ptr<MeshFunction<unsigned int> >
    facet_orientation = mesh.data().mesh_function("facet_orientation");
  if (facet_orientation && facet_orientation->dim() != D - 1)
  {
    dolfin_error("Assembler.cpp",
                 "assemble form over interior facets",
                 "Expecting facet orientation to be defined on facets (not dimension %d)",
                 facet_orientation->dim());
  }

  // Assemble over interior facets (the facets of the mesh)
  Progress p(AssemblerTools::progress_message(A.rank(), "interior facets"),
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
    if (interior_facet_domains && interior_facet_domains->size() > 0)
    {
      const uint domain = (*interior_facet_domains)[*facet];
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
    for (uint i = 0; i < form_rank; ++i)
    {
      // Get dofs for each cell
      const std::vector<uint>& cell_dofs0 = dofmaps[i]->cell_dofs(cell0.index());
      const std::vector<uint>& cell_dofs1 = dofmaps[i]->cell_dofs(cell1.index());

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

    // Apply boundary conditions
    apply_local_bc(ufc.macro_A, ufc_nonsymm.macro_A, macro_dof_ptrs);

    // Add entries to global tensor
    A.add(&ufc.macro_A[0], macro_dofs);
    A_nonsymm.add(&ufc_nonsymm.macro_A[0], macro_dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void SymmetricAssembler::PImpl::apply_local_bc(std::vector<double> &local_A,
                                               std::vector<double> &local_A_nonsymm,
                                               const std::vector<const std::vector<uint>*> &dofs)
{
  // Get local dimensions
  const uint num_local_rows = dofs[0]->size();
  const uint num_local_cols = dofs[1]->size();
  const uint num_entries = num_local_rows*num_local_cols;

  // Initialize the nonsymmetric part to 0.0
  zerofill(local_A_nonsymm);

  // Convenience aliases
  const std::vector<uint> &row_dofs = *dofs[0];
  const std::vector<uint> &col_dofs = *dofs[1];

  if (matching_bcs && row_dofs!=col_dofs)
    dolfin_error("SymmetricAssembler.cpp",
                 "apply_local_bc",
                 "Same BCs are used for rows and columns, but dofmaps don't match");

  // Store the local boundary conditions, to avoid having to search for them
  // twice in the (common) case of matching_bcs.
  local_row_is_bc.resize(num_local_rows);
  for (uint row = 0; row < num_local_rows; ++row)
  {
    DirichletBC::Map::const_iterator bc_item = row_bc_values.find(row_dofs[row]);
    local_row_is_bc[row] = (bc_item != row_bc_values.end());
  }

  // Zero BC rows in local_A
  for (uint row = 0; row < num_local_rows; ++row)
    if (local_row_is_bc[row])
      zerofill(&local_A[row*num_local_cols], num_local_cols);

  for (uint col = 0; col < num_local_cols; ++col)
  {
    // Do nothing if column is not affected by BCs
    if (matching_bcs) {
      if (!local_row_is_bc[col])
        continue;
    }
    else
    {
      DirichletBC::Map::const_iterator bc_item = col_bc_values.find(col_dofs[col]);
      if (bc_item == col_bc_values.end())
        continue;
    }

    // The column is affected by BCs, so move it to A_nonsymm and zero it in A
    for (uint entry = col; entry < num_entries; entry += num_local_cols)
    {
      local_A_nonsymm[entry] = local_A[entry];
      local_A[entry] = 0.0;
    }

    // Set diagonal to 1.0, IF the matrix is a diagonal block (i.e, if the bcs
    // match). Do this only once per dof, even if it is part of multiple cells.
    if (matching_bcs)
    {
      const bool diagonal_already_set = !already_set_diagonals.insert(col_dofs[col]).second;
      if (!diagonal_already_set)
        local_A[col+col*num_local_cols] = 1.0;
    }
  }
}
