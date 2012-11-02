// Copyright (C) 2012 Joachim B. Haga
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
// First added:  2012-02-01
// Last changed: 2012-10-04

#include <dolfin/common/utils.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/parameter/GlobalParameters.h>
#include "Form.h"
#include "UFC.h"
#include "SymmetricAssembler.h"

using namespace dolfin;

class SymmetricAssembler::Private
{
  public:

  // Parameters that need to be accessible from add_to_global_tensor. These
  // are stored (by the assemble() method) as instance variables, since the
  // add_to_global_tensor interface is fixed.

  Private(GenericMatrix& _B) : B(_B) {}

  GenericMatrix &B; // The non-symmetric-part coefficient matrix

  bool matching_bcs;    // true if row_bcs==col_bcs
  DirichletBC::Map row_bc_values; // derived from row_bcs
  DirichletBC::Map col_bc_values; // derived from col_bcs, but empty if matching_bcs

  // These are used to keep track of which diagonals (BCs) have been set:
  std::pair<uint,uint> processor_dof_range;
  std::set<uint> inserted_diagonals;

  // Scratch variables
  std::vector<double> local_B;
  std::vector<bool> local_row_is_bc;

};
//-----------------------------------------------------------------------------
void SymmetricAssembler::assemble(GenericMatrix& A,
                                  GenericMatrix& B,
                                  const Form& a,
                                  const std::vector<const DirichletBC*> row_bcs,
                                  const std::vector<const DirichletBC*> col_bcs,
                                  const SubDomain &sub_domain)
{
  //
  // Convert the SubDomain to meshfunctions, suitable for the real assemble
  // method. Copied from Assembler, except for the final assemble(...) call.
  //

  dolfin_assert(a.ufc_form());

  // Extract mesh
  const Mesh& mesh = a.mesh();

  // Extract cell domains
  boost::scoped_ptr<MeshFunction<uint> > cell_domains;
  if (a.ufc_form()->num_cell_domains() > 0)
  {
    cell_domains.reset(new MeshFunction<uint>(mesh, mesh.topology().dim(), 1));
    sub_domain.mark(*cell_domains, 0);
  }

  // Extract facet domains
  boost::scoped_ptr<MeshFunction<uint> > facet_domains;
  if (a.ufc_form()->num_exterior_facet_domains() > 0 ||
      a.ufc_form()->num_interior_facet_domains() > 0)
  {
    facet_domains.reset(new MeshFunction<uint>(mesh, mesh.topology().dim() - 1, 1));
    sub_domain.mark(*facet_domains, 0);
  }

  // Assemble
  assemble(A, B, a, row_bcs, col_bcs, cell_domains.get(), facet_domains.get(), facet_domains.get());
  
  // Periodic modification
  if (!mesh.facet_pairs.empty())
  {
      A.ident_zeros();
  }
}
//-----------------------------------------------------------------------------
void SymmetricAssembler::assemble(GenericMatrix& A,
                                  GenericMatrix& B,
                                  const Form& a,
                                  const std::vector<const DirichletBC*> row_bcs,
                                  const std::vector<const DirichletBC*> col_bcs,
                                  const MeshFunction<uint>* cell_domains,
                                  const MeshFunction<uint>* exterior_facet_domains,
                                  const MeshFunction<uint>* interior_facet_domains)
{
  dolfin_assert(a.rank() == 2);

  #ifdef HAS_OPENMP
  const uint num_threads = parameters["num_threads"];
  if (num_threads > 0)
  {
    dolfin_error("SymmetricAssembler.cpp", "assemble",
                 "OpenMP symmetric assemble not supported");
  }
  #endif

  // Store parameters that the standard assembler don't use
  impl = new Private(B);

  impl->matching_bcs = (row_bcs == col_bcs);

  // Get Dirichlet dofs rows and values for local mesh
  for (uint i = 0; i < row_bcs.size(); ++i)
  {
    row_bcs[i]->get_boundary_values(impl->row_bc_values);
    if (MPI::num_processes() > 1 && row_bcs[i]->method() != "pointwise")
      row_bcs[i]->gather(impl->row_bc_values);
  }
  if (!impl->matching_bcs)
  {
    // Get Dirichlet dofs columns and values for local mesh
    for (uint i = 0; i < col_bcs.size(); ++i)
    {
      col_bcs[i]->get_boundary_values(impl->col_bc_values);
      if (MPI::num_processes() > 1 && col_bcs[i]->method() != "pointwise")
        col_bcs[i]->gather(impl->col_bc_values);
    }
  }

  // Initialize the global nonsymmetric tensor (the symmetric one is handled by Assembler)
  const std::vector<std::pair<std::pair<uint, uint>, std::pair<uint, uint> > > periodic_master_slave_dofs;
  init_global_tensor(impl->B, a, periodic_master_slave_dofs);

  // Get dofs that are local to this processor
  impl->processor_dof_range = impl->B.local_range(0);

  // Call the standard assembler (which in turn calls add_to_global_tensor)
  Assembler::assemble(A, a, cell_domains, exterior_facet_domains, interior_facet_domains);

  // Finalize assembly of global nonsymmetric tensor (the symmetric one is
  // finalized by Assembler)
  if (finalize_tensor)
  {
    impl->B.apply("add");
  }

  // Delete the private instance holding additional parameters for
  // add_to_global_tensor
  delete impl;
}
//-----------------------------------------------------------------------------
void SymmetricAssembler::add_to_global_tensor(GenericTensor &A,
                                              std::vector<double>& local_A,
                                              std::vector<const std::vector<uint>* >& dofs)
{
  // Apply boundary conditions, and move affected columns of the local element
  // tensor, to restore symmetry.

  // Get local dimensions
  const uint num_local_rows = dofs[0]->size();
  const uint num_local_cols = dofs[1]->size();

  // Return value, true if columns have been moved from local_A to local_B
  bool local_B_is_set = false;

  // Convenience aliases
  const std::vector<uint>& row_dofs = *dofs[0];
  const std::vector<uint>& col_dofs = *dofs[1];

  if (impl->matching_bcs && row_dofs != col_dofs)
    dolfin_error("SymmetricAssembler.cpp",
                 "make_bc_symmetric",
                 "Same BCs are used for rows and columns, but dofmaps don't match");

  // Store the local boundary conditions, to avoid multiple searches in the
  // (common) case of matching_bcs
  impl->local_row_is_bc.resize(num_local_rows);
  for (uint row = 0; row < num_local_rows; ++row)
  {
    DirichletBC::Map::const_iterator bc_item = impl->row_bc_values.find(row_dofs[row]);
    impl->local_row_is_bc[row] = (bc_item != impl->row_bc_values.end());
  }

  // Clear matrix rows belonging to BCs. These modifications destroy symmetry.
  for (uint row = 0; row < num_local_rows; ++row)
  {
    // Do nothing if row is not affected by BCs
    if (!impl->local_row_is_bc[row])
      continue;

    // Zero the row in local_A
    zerofill(&local_A[row*num_local_cols], num_local_cols);

    // Set the diagonal, if we're in a diagonal block...
    if (impl->matching_bcs)
    {
      // ...but only set it on the owning processor
      const uint dof = row_dofs[row];
      if (dof >= impl->processor_dof_range.first && dof < impl->processor_dof_range.second)
      {
        // ...and only once.
        const bool already_inserted = !impl->inserted_diagonals.insert(dof).second;
        if (!already_inserted)
          local_A[row + row*num_local_cols] = 1.0;
      }
    }
  }

  // Modify matrix columns belonging to BCs. These modifications restore
  // symmetry, but the entries must be moved to the asymm matrix instead of
  // just cleared.
  for (uint col = 0; col < num_local_cols; ++col)
  {
    // Do nothing if column is not affected by BCs
    if (impl->matching_bcs) {
      if (!impl->local_row_is_bc[col])
        continue;
    }
    else
    {
      DirichletBC::Map::const_iterator bc_item = impl->col_bc_values.find(col_dofs[col]);
      if (bc_item == impl->col_bc_values.end())
        continue;
    }

    // Resize and zero the local asymmetric part before use. The resize is a
    // no-op most of the time.
    if (!local_B_is_set)
    {
      impl->local_B.resize(local_A.size());
      zerofill(impl->local_B);
      local_B_is_set = true;
    }

    // Move the column to B, zero it in A
    for (uint row = 0; row < num_local_rows; ++row)
    {
      if (!impl->local_row_is_bc[row])
      {
        const uint entry = col + row*num_local_cols;
        impl->local_B[entry] = local_A[entry];
        local_A[entry] = 0.0;
      }
    }
  }

  // Add entries to global tensor.
  A.add(&local_A[0], dofs);
  if (local_B_is_set)
  {
    impl->B.add(&impl->local_B[0], dofs);
  }
}
//-----------------------------------------------------------------------------
