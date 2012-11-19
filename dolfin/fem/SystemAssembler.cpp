// Copyright (C) 2008-2012 Kent-Andre Mardal and Garth N. Wells
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
// Modified by Anders Logg 2008-2011
// Modified by Joachim B Haga 2012
//
// First added:  2009-06-22
// Last changed: 2012-11-17

#include <armadillo>
#include <dolfin/common/Timer.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/SubDomain.h>
#include "AssemblerBase.h"
#include "DirichletBC.h"
#include "FiniteElement.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "UFC.h"
#include "SystemAssembler.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void SystemAssembler::assemble(GenericMatrix& A, GenericVector& b,
                               const Form& a, const Form& L)
{
  std::vector<const DirichletBC*> bcs;
  assemble(A, b, a, L, bcs, 0, 0, 0, 0);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(GenericMatrix& A, GenericVector& b,
                               const Form& a, const Form& L,
                               const DirichletBC& bc)
{
  std::vector<const DirichletBC*> bcs;
  bcs.push_back(&bc);
  assemble(A, b, a, L, bcs, 0, 0, 0, 0);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(GenericMatrix& A, GenericVector& b,
                               const Form& a, const Form& L,
                               const std::vector<const DirichletBC*> bcs)
{
  assemble(A, b, a, L, bcs, 0, 0, 0, 0);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(GenericMatrix& A, GenericVector& b,
                               const Form& a, const Form& L,
                               const std::vector<const DirichletBC*> bcs,
                               const MeshFunction<std::size_t>* cell_domains,
                               const MeshFunction<std::size_t>* exterior_facet_domains,
                               const MeshFunction<std::size_t>* interior_facet_domains,
                               const GenericVector* x0)
{
  // Set timer
  Timer timer("Assemble system");

  // Get mesh
  const Mesh& mesh = a.mesh();
  dolfin_assert(mesh.ordered());

  // Get cell domains
  if (!cell_domains || cell_domains->empty())
  {
    if (a.cell_domains_shared_ptr() || L.cell_domains_shared_ptr())
    {
      if (a.cell_domains_shared_ptr() && L.cell_domains_shared_ptr())
      {
        if (a.cell_domains_shared_ptr() != L.cell_domains_shared_ptr())
          warning("Bilinear and linear form must have same attached cell subdomains in SystemAssembler.");
      }
      else if (a.cell_domains_shared_ptr())
        cell_domains = a.cell_domains_shared_ptr().get();
      else
        cell_domains = L.cell_domains_shared_ptr().get();

      if (mesh.domains().facet_domains(mesh))
        warning("Ignoring cell domains defined as part of mesh in system assembler.");
    }
    else
      cell_domains = mesh.domains().cell_domains(mesh).get();
  }

  // Get exterior facet domains
  if (!exterior_facet_domains || exterior_facet_domains->empty())
  {
    if (a.exterior_facet_domains_shared_ptr() || L.exterior_facet_domains_shared_ptr())
    {
      if (a.exterior_facet_domains_shared_ptr() && L.exterior_facet_domains_shared_ptr())
      {
        if (a.exterior_facet_domains_shared_ptr() != L.exterior_facet_domains_shared_ptr())
          warning("Bilinear and linear form must have same attached exterior facet subdomains in SystemAssembler.");
      }
      else if (a.exterior_facet_domains_shared_ptr())
        exterior_facet_domains = a.exterior_facet_domains_shared_ptr().get();
      else
        exterior_facet_domains = L.exterior_facet_domains_shared_ptr().get();

      if (mesh.domains().facet_domains(mesh))
        warning("Ignoring exterior facet domains defined as part of mesh in system assembler.");
    }
    else
      exterior_facet_domains = mesh.domains().facet_domains(mesh).get();
  }

  // Get interior facet domains
  if (!interior_facet_domains || interior_facet_domains->empty())
  {
    if (a.interior_facet_domains_shared_ptr() || L.interior_facet_domains_shared_ptr())
    {
      if (a.interior_facet_domains_shared_ptr() && L.interior_facet_domains_shared_ptr())
      {
        if (a.interior_facet_domains_shared_ptr() != L.interior_facet_domains_shared_ptr())
          warning("Bilinear and linear form must have same attached interior facet subdomains in SystemAssembler.");
      }
      else if (a.interior_facet_domains_shared_ptr())
        interior_facet_domains = a.interior_facet_domains_shared_ptr().get();
      else
        interior_facet_domains = L.interior_facet_domains_shared_ptr().get();

      if (mesh.domains().facet_domains(mesh))
        warning("Ignoring interior facet domains defined as part of mesh in system assembler.");
    }
    else
      interior_facet_domains = mesh.domains().facet_domains(mesh).get();
  }

  // Check forms
  AssemblerBase::check(a);
  AssemblerBase::check(L);

  // Check that we have a bilinear and a linear form
  dolfin_assert(a.rank() == 2);
  dolfin_assert(L.rank() == 1);

  // Check that forms share a function space
  if (*a.function_space(1) != *L.function_space(0))
  {
    dolfin_error("SystemAssembler.cpp",
		 "assemble system",
		 "expected forms (a, L) to share a FunctionSpace");
  }

  // FIXME: This may update coefficients twice. Checked for shared coefficients

  // Update off-process coefficients for a
  std::vector<boost::shared_ptr<const GenericFunction> > coefficients = a.coefficients();
  for (uint i = 0; i < coefficients.size(); ++i)
    coefficients[i]->update();

  // Update off-process coefficients for L
  coefficients = L.coefficients();
  for (uint i = 0; i < coefficients.size(); ++i)
    coefficients[i]->update();

  // Create data structures for local assembly data
  UFC A_ufc(a), b_ufc(L);

  // Initialize global tensors
  const std::vector<std::pair<std::pair<std::size_t, std::size_t>, std::pair<std::size_t, std::size_t> > > periodic_master_slave_dofs;
  init_global_tensor(A, a, periodic_master_slave_dofs);
  init_global_tensor(b, L, periodic_master_slave_dofs);

  // Allocate data
  Scratch data(a, L);

  // Get Dirichlet dofs and values for local mesh
  DirichletBC::Map boundary_values;
  for (std::size_t i = 0; i < bcs.size(); ++i)
  {
    bcs[i]->get_boundary_values(boundary_values);
    if (MPI::num_processes() > 1 && bcs[i]->method() != "pointwise")
      bcs[i]->gather(boundary_values);
  }

  // Modify boundary values for incremental (typically nonlinear) problems
  if (x0)
  {
    if (MPI::num_processes() > 1)
      warning("Parallel symmetric assembly over interior facets for nonlinear problems is untested");

    dolfin_assert(x0->size() == a.function_space(1)->dofmap()->global_dimension());

    const std::size_t num_bc_dofs = boundary_values.size();
    std::vector<DolfinIndex> bc_indices;
    std::vector<double> bc_values;
    bc_indices.reserve(num_bc_dofs);
    bc_values.reserve(num_bc_dofs);

    // Build list of boundary dofs and values
    DirichletBC::Map::const_iterator bv;
    for (bv = boundary_values.begin(); bv != boundary_values.end(); ++bv)
    {
      bc_indices.push_back(bv->first);
      bc_values.push_back(bv->second);
    }

    // Modify bc values
    std::vector<double> x0_values(num_bc_dofs);
    x0->get_local(&x0_values[0], num_bc_dofs, &bc_indices[0]);
    for (std::size_t i = 0; i < num_bc_dofs; i++)
      bc_values[i] = x0_values[i] - bc_values[i];
  }

  // Check whether we should do cell-wise or facet-wise assembly
  if (A_ufc.form.num_interior_facet_domains() == 0 &&
      b_ufc.form.num_interior_facet_domains() == 0)
  {
    // Assemble cell-wise (no interior facet integrals)
    cell_wise_assembly(A, b, a, L,
                       A_ufc, b_ufc,
                       data, boundary_values,
                       cell_domains,
                       exterior_facet_domains);
  }
  else
  {
    // Facet-wise assembly is not working in parallel
    not_working_in_parallel("System assembly over interior facets");

    // Facet-wise assembly does not support subdomains
    if (A_ufc.form.num_cell_domains() > 1 ||
        b_ufc.form.num_cell_domains() > 1 ||
        A_ufc.form.num_exterior_facet_domains() > 1 ||
        b_ufc.form.num_exterior_facet_domains() > 1 ||
        A_ufc.form.num_interior_facet_domains() > 1 ||
        b_ufc.form.num_interior_facet_domains() > 1)
    {
      dolfin_error("SystemAssembler.cpp",
                   "assemble system",
                   "System assembler does not support subdomains forms containing "
                   "interior facet integrals");
    }

    // Assemble facet-wise (including cell assembly)
    facet_wise_assembly(A, b, a, L,
                        A_ufc, b_ufc,
                        data, boundary_values,
                        cell_domains,
                        exterior_facet_domains,
                        interior_facet_domains);
  }

  // Finalise assembly
  if (finalize_tensor)
  {
    A.apply("add");
    b.apply("add");
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::cell_wise_assembly(GenericMatrix& A, GenericVector& b,
                                         const Form& a, const Form& L,
                                         UFC& A_ufc, UFC& b_ufc, Scratch& data,
                                         const DirichletBC::Map& boundary_values,
                                         const MeshFunction<std::size_t>* cell_domains,
                                         const MeshFunction<std::size_t>* exterior_facet_domains)
{
  // FIXME: We can used some std::vectors or array pointers for the A and b
  // related terms to cut down on code repetition.

  const Mesh& mesh = a.mesh();
  dolfin_assert(mesh.ordered());

  // Form ranks
  const uint a_rank = a.rank();
  const uint L_rank = L.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> a_dofmaps;
  for (uint i = 0; i < a_rank; ++i)
    a_dofmaps.push_back(a.function_space(i)->dofmap().get());

  std::vector<const GenericDofMap*> L_dofmaps;
  for (uint i = 0; i < L_rank; ++i)
    L_dofmaps.push_back(L.function_space(i)->dofmap().get());

  // Vector to hold dof map for a cell
  std::vector<const std::vector<DolfinIndex>* > a_dofs(a_rank);
  std::vector<const std::vector<DolfinIndex>* > L_dofs(L_rank);

  // Create pointers to hold integral objects
  const ufc::cell_integral* A_cell_integral(0);
  const ufc::cell_integral* b_cell_integral(0);
  const ufc::exterior_facet_integral* A_exterior_facet_integral(0);
  const ufc::exterior_facet_integral* b_exterior_facet_integral(0);

  // Initialize integrals to first integral. These will be overwritten
  // below if subdomains have been specified. Otherwise, the same integral
  // will be used throughout the whole mesh
  if (A_ufc.form.num_cell_domains() > 0)
    A_cell_integral = A_ufc.cell_integrals[0].get();
  if (b_ufc.form.num_cell_domains() > 0)
    b_cell_integral = b_ufc.cell_integrals[0].get();
  if (A_ufc.form.num_exterior_facet_domains() > 0)
    A_exterior_facet_integral = A_ufc.exterior_facet_integrals[0].get();
  if (b_ufc.form.num_exterior_facet_domains() > 0)
    b_exterior_facet_integral = b_ufc.exterior_facet_integrals[0].get();

  // If using external facet integrals
  if (A_ufc.form.num_exterior_facet_domains() > 0 ||
      b_ufc.form.num_exterior_facet_domains() > 0)
  {
    // Compute facets and facet - cell connectivity if not already computed
    const uint D = mesh.topology().dim();
    mesh.init(D - 1);
    mesh.init(D - 1, D);
  }

  // Iterate over all cells
  Progress p("Assembling system (cell-wise)", mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Reset cell tensor and vector
    std::fill(data.Ae.begin(), data.Ae.end(), 0.0);
    std::fill(data.be.begin(), data.be.end(), 0.0);

    // Get cell integrals for sub domain (if any)
    if (cell_domains && !cell_domains->empty())
    {
      const std::size_t domain = (*cell_domains)[*cell];
      if (domain < A_ufc.form.num_cell_domains())
        A_cell_integral = A_ufc.cell_integrals[domain].get();
      else
        A_cell_integral = 0;
      if (domain < b_ufc.form.num_cell_domains())
        b_cell_integral = b_ufc.cell_integrals[domain].get();
      else
        b_cell_integral = 0;
    }

    // Compute cell tensor for A
    if (A_cell_integral)
    {
      // Update to current cell
      A_ufc.update(*cell);

      // Tabulate cell tensor
      A_cell_integral->tabulate_tensor(&A_ufc.A[0], A_ufc.w(), A_ufc.cell);
      for (uint i = 0; i < data.Ae.size(); ++i)
        data.Ae[i] += A_ufc.A[i];
    }

    // Compute cell tensor for b
    if (b_cell_integral)
    {
      // Update to current cell
      b_ufc.update(*cell);

      // Tabulate cell tensor
      b_cell_integral->tabulate_tensor(&b_ufc.A[0], b_ufc.w(), b_ufc.cell);
      for (uint i = 0; i < data.be.size(); ++i)
        data.be[i] += b_ufc.A[i];
    }

    // Compute exterior facet integral if present
    if (A_ufc.form.num_exterior_facet_domains() > 0 ||
        b_ufc.form.num_exterior_facet_domains() > 0)
    {
      for (FacetIterator facet(*cell); !facet.end(); ++facet)
      {
        // Only consider exterior facets
        if (!facet->exterior())
          continue;

        // Get exterior facet integrals for sub domain (if any)
        if (exterior_facet_domains && !exterior_facet_domains->empty())
        {
          const std::size_t domain = (*exterior_facet_domains)[*facet];
          if (domain < A_ufc.form.num_exterior_facet_domains())
            A_exterior_facet_integral = A_ufc.exterior_facet_integrals[domain].get();
          else
            A_exterior_facet_integral = 0;
          if (domain < b_ufc.form.num_exterior_facet_domains())
            b_exterior_facet_integral = b_ufc.exterior_facet_integrals[domain].get();
          else
            b_exterior_facet_integral = 0;
        }

        // Skip if there are no integrals
        if (!A_exterior_facet_integral && !b_exterior_facet_integral)
          continue;

        // Extract local facet index
        const uint local_facet = cell->index(*facet);

        // Add exterior facet tensor for A
        if (A_exterior_facet_integral)
        {
          // Update to current cell
          A_ufc.update(*cell, local_facet);

          // Tabulate exterior facet tensor
          A_exterior_facet_integral->tabulate_tensor(&A_ufc.A[0],
                                                     A_ufc.w(),
                                                     A_ufc.cell,
                                                     local_facet);
          for (uint i = 0; i < data.Ae.size(); i++)
            data.Ae[i] += A_ufc.A[i];
        }

        // Add exterior facet tensor for b
        if (b_exterior_facet_integral)
        {
          // Update to current cell
          b_ufc.update(*cell, local_facet);

          // Tabulate exterior facet tensor
          b_exterior_facet_integral->tabulate_tensor(&b_ufc.A[0],
                                                     b_ufc.w(),
                                                     b_ufc.cell,
                                                     local_facet);
          for (uint i = 0; i < data.be.size(); i++)
            data.be[i] += b_ufc.A[i];
        }
      }
    }

    // Get local-to-global dof maps for cell
    a_dofs[0] = &(a_dofmaps[0]->cell_dofs(cell->index()));
    a_dofs[1] = &(a_dofmaps[1]->cell_dofs(cell->index()));
    L_dofs[0] = &(L_dofmaps[0]->cell_dofs(cell->index()));

    dolfin_assert(L_dofs[0] == a_dofs[1]);

    // Modify local matrix/element for Dirichlet boundary conditions
    apply_bc(&(data.Ae)[0], &(data.be)[0], boundary_values, a_dofs);

    // Add entries to global tensor
    A.add(&data.Ae[0], a_dofs);
    b.add(&data.be[0], L_dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::facet_wise_assembly(GenericMatrix& A, GenericVector& b,
                              const Form& a, const Form& L,
                              UFC& A_ufc, UFC& b_ufc, Scratch& data,
                              const DirichletBC::Map& boundary_values,
                              const MeshFunction<std::size_t>* cell_domains,
                              const MeshFunction<std::size_t>* exterior_facet_domains,
                              const MeshFunction<std::size_t>* interior_facet_domains)
{
  // Extract mesh and coefficients
  const Mesh& mesh = a.mesh();
  const std::vector<boost::shared_ptr<const GenericFunction> > A_coefficients = a.coefficients();
  const std::vector<boost::shared_ptr<const GenericFunction> > b_coefficients = L.coefficients();

  // Compute facets and facet - cell connectivity if not already computed
  const uint D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);

  // Form ranks
  const uint a_rank = a.rank();
  const uint L_rank = L.rank();

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> a_dofmaps;
  for (uint i = 0; i < a_rank; ++i)
    a_dofmaps.push_back(a.function_space(i)->dofmap().get());

  std::vector<const GenericDofMap*> L_dofmaps;
  for (uint i = 0; i < L_rank; ++i)
    L_dofmaps.push_back(L.function_space(i)->dofmap().get());

  // Vector to hold dof map for a cell
  std::vector<const std::vector<DolfinIndex>* > a_dofs(a_rank), L_dofs(L_rank);

  // Iterate over facets
  Progress p("Assembling system (facet-wise)", mesh.num_facets());
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Interior facet
    if (facet->num_entities(mesh.topology().dim()) == 2)
    {
      // Get cells incident with facet and update UFC objects
      Cell cell0(mesh, facet->entities(mesh.topology().dim())[0]);
      Cell cell1(mesh, facet->entities(mesh.topology().dim())[1]);

      // Get local facet index
      const uint local_facet0 = cell0.index(*facet);
      const uint local_facet1 = cell1.index(*facet);

      A_ufc.update(cell0, local_facet0, cell1, local_facet1);
      b_ufc.update(cell0, local_facet0, cell1, local_facet1);

      // Reset some temp data
      std::fill(A_ufc.macro_A.begin(), A_ufc.macro_A.end(), 0.0);
      std::fill(b_ufc.macro_A.begin(), b_ufc.macro_A.end(), 0.0);

      // Assemble interior facet and neighbouring cells if needed
      assemble_interior_facet(A, b, A_ufc, b_ufc, a, L, cell0, cell1, *facet,
                              data, boundary_values);
    }
    else // Exterior facet
    {
      // Get mesh cell to which mesh facet belongs (pick first, there is only one)
      Cell cell(mesh, facet->entities(mesh.topology().dim())[0]);

      // Get local index of facet with respect to the cell
      const uint local_facet = cell.index(*facet);

      A_ufc.update(cell, local_facet);
      b_ufc.update(cell, local_facet);

      // Reset some temp data
      std::fill(A_ufc.A.begin(), A_ufc.A.end(), 0.0);
      std::fill(b_ufc.A.begin(), b_ufc.A.end(), 0.0);

      // Initialize macro element matrix/vector to zero
      data.zero_cell();

      // Assemble exterior facet and attached cells if needed
      assemble_exterior_facet(A, b, A_ufc, b_ufc, a, L, cell, *facet, data,
                              boundary_values);
    }
    p++;
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::compute_tensor_on_one_interior_facet(const Form& a,
            UFC& ufc, const Cell& cell0, const Cell& cell1, const Facet& facet,
            const MeshFunction<std::size_t>* interior_facet_domains)
{
  const std::vector<boost::shared_ptr<const GenericFunction> > coefficients = a.coefficients();

  // Facet integral
  ufc::interior_facet_integral* interior_facet_integral = ufc.interior_facet_integrals[0].get();

  // Get integral for sub domain (if any)
  if (interior_facet_domains && !interior_facet_domains->empty())
  {
    const std::size_t domain = (*interior_facet_domains)[facet];
    if (domain < ufc.form.num_interior_facet_domains())
      interior_facet_integral = ufc.interior_facet_integrals[domain].get();
  }

  // Get local index of facet with respect to each cell
  const uint local_facet0 = cell0.index(facet);
  const uint local_facet1 = cell1.index(facet);

  // Update to current pair of cells
  ufc.update(cell0, local_facet0, cell1, local_facet1);

  interior_facet_integral->tabulate_tensor(&ufc.macro_A[0], ufc.macro_w(),
                                           ufc.cell0, ufc.cell1,
                                           local_facet0, local_facet1);
}
//-----------------------------------------------------------------------------
inline void SystemAssembler::apply_bc(double* A, double* b,
                     const DirichletBC::Map& boundary_values,
                     const std::vector<const std::vector<DolfinIndex>* >& global_dofs)
{
  // Wrap matrix and vector as Armadillo. Armadillo matrix storgae is
  // column-major, so all operations are transposed.
  arma::mat _A(A, global_dofs[1]->size(), global_dofs[0]->size(), false, true);
  arma::rowvec _b(b, global_dofs[0]->size(), false, true);

  //bool bc_applied = false;
  // Loop over rows
  for (std::size_t i = 0; i < _A.n_rows; ++i)
  {
    const std::size_t ii = (*global_dofs[1])[i];
    DirichletBC::Map::const_iterator bc_value = boundary_values.find(ii);
    if (bc_value != boundary_values.end())
    {
      // Zero row
      _A.unsafe_col(i).fill(0.0);

      // Modify RHS (subtract (bc_column(A))*bc_val from b)
      _b -= _A.row(i)*bc_value->second;

      // Get measure of size of RHS components
      const double b_norm = arma::norm(_b, 1)/_b.size();

      // Zero column
      _A.row(i).fill(0.0);

      // Place 1 on diagonal and bc on RHS (i th row ). Rescale to avoid
      // distortion of RHS norm.
      if (std::abs(bc_value->second) < (b_norm + DOLFIN_EPS))
      {
        _b(i)    = bc_value->second;
        _A(i, i) = 1.0;
      }
      else
      {
        dolfin_assert(std::abs(bc_value->second) > 0.0);
        _b(i)    = b_norm;
        _A(i, i) = b_norm/bc_value->second;
      }
    }
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble_interior_facet(GenericMatrix& A, GenericVector& b,
                                              UFC& A_ufc, UFC& b_ufc,
                                              const Form& a, const Form& L,
                                              const Cell& cell0, const Cell& cell1,
                                              const Facet& facet, Scratch& data,
                                              const DirichletBC::Map& boundary_values)
{
  // Facet orientation not supported
  if (cell0.mesh().data().mesh_function("facet_orientation"))
  {
    dolfin_error("SystemAssembler.cpp",
                 "assemble system",
                 "User-defined facet orientation is not supported by system assembler");
  }

  const std::size_t cell0_index = cell0.index();
  const std::size_t cell1_index = cell1.index();

  // Tabulate dofs
  const std::vector<DolfinIndex>& a0_dofs0 = a.function_space(0)->dofmap()->cell_dofs(cell0_index);
  const std::vector<DolfinIndex>& a1_dofs0 = a.function_space(1)->dofmap()->cell_dofs(cell0_index);
  const std::vector<DolfinIndex>& L_dofs0  = L.function_space(0)->dofmap()->cell_dofs(cell0_index);

  const std::vector<DolfinIndex>& a0_dofs1 = a.function_space(0)->dofmap()->cell_dofs(cell1_index);
  const std::vector<DolfinIndex>& a1_dofs1 = a.function_space(1)->dofmap()->cell_dofs(cell1_index);
  const std::vector<DolfinIndex>& L_dofs1  = L.function_space(0)->dofmap()->cell_dofs(cell1_index);

  // Cell integrals
  const ufc::cell_integral* A_cell_integral = A_ufc.cell_integrals[0].get();
  const ufc::cell_integral* b_cell_integral = b_ufc.cell_integrals[0].get();

  // Compute facet contribution to A
  if (A_ufc.form.num_interior_facet_domains() > 0)
    compute_tensor_on_one_interior_facet(a, A_ufc, cell0, cell1, facet, 0);

  // Compute facet contribution to
  if (b_ufc.form.num_interior_facet_domains() > 0)
    compute_tensor_on_one_interior_facet(L, b_ufc, cell0, cell1, facet, 0);

  // Get local facet index
  const uint facet0 = cell0.index(facet);
  const uint facet1 = cell1.index(facet);

  // If we have local facet 0, compute cell contribution
  if (facet0 == 0)
  {
    if (A_ufc.form.num_cell_domains() > 0)
    {
      A_ufc.update(cell0);
      A_cell_integral->tabulate_tensor(&A_ufc.A[0], A_ufc.w(), A_ufc.cell);
      const std::size_t nn = a0_dofs0.size();
      const std::size_t mm = a1_dofs0.size();
      for (std::size_t i = 0; i < mm; i++)
        for (std::size_t j = 0; j < nn; j++)
          A_ufc.macro_A[2*i*nn+j] += A_ufc.A[i*nn+j];
    }

    if (b_ufc.form.num_cell_domains() > 0)
    {
      b_ufc.update(cell0);
      b_cell_integral->tabulate_tensor(&b_ufc.A[0], b_ufc.w(), b_ufc.cell);
      for (std::size_t i = 0; i < L_dofs0.size(); i++)
        b_ufc.macro_A[i] += b_ufc.A[i];
    }
  }

  // If we have local facet 0, compute and add cell contribution
  if (facet1 == 0)
  {
    if (A_ufc.form.num_cell_domains() > 0)
    {
      A_ufc.update(cell1);

      A_cell_integral->tabulate_tensor(&A_ufc.A[0], A_ufc.w(), A_ufc.cell);
      const std::size_t nn = a0_dofs1.size();
      const std::size_t mm = a1_dofs1.size();
      for (std::size_t i = 0; i < mm; i++)
        for (std::size_t j = 0; j < nn; j++)
          A_ufc.macro_A[2*nn*mm + 2*i*nn + nn + j] += A_ufc.A[i*nn+j];
    }

    if (b_ufc.form.num_cell_domains() > 0)
    {
      b_ufc.update(cell1);

      b_cell_integral->tabulate_tensor(&b_ufc.A[0], b_ufc.w(), b_ufc.cell);
      for (std::size_t i = 0; i < L_dofs0.size(); i++)
        b_ufc.macro_A[L_dofs0.size() + i] += b_ufc.A[i];
    }
  }

  // Vector to hold dofs for cells
  std::vector<std::vector<DolfinIndex> > a_macro_dofs(2);
  std::vector<std::vector<DolfinIndex> > L_macro_dofs(1);

  // Resize dof vector
  a_macro_dofs[0].resize(a0_dofs0.size() + a0_dofs1.size());
  a_macro_dofs[1].resize(a1_dofs0.size() + a1_dofs1.size());
  L_macro_dofs[0].resize(L_dofs0.size() + L_dofs1.size());

  // Tabulate dofs for each dimension on macro element
  std::copy(a0_dofs0.begin(), a0_dofs0.end(), a_macro_dofs[0].begin());
  std::copy(a0_dofs1.begin(), a0_dofs1.end(),
            a_macro_dofs[0].begin() + a0_dofs0.size());

  std::copy(a1_dofs0.begin(), a1_dofs0.end(), a_macro_dofs[1].begin());
  std::copy(a1_dofs1.begin(), a1_dofs1.end(),
            a_macro_dofs[1].begin() + a1_dofs0.size());

  std::copy(L_dofs0.begin(), L_dofs0.end(), L_macro_dofs[0].begin());
  std::copy(L_dofs1.begin(), L_dofs1.end(),
            L_macro_dofs[0].begin() + L_dofs0.size());

  // Modify local matrix/element for Dirichlet boundary conditions
  std::vector<const std::vector<DolfinIndex>* > _a_macro_dofs(2);
  _a_macro_dofs[0] = &a_macro_dofs[0];
  _a_macro_dofs[1] = &a_macro_dofs[1];

  apply_bc(&A_ufc.macro_A[0], &b_ufc.macro_A[0], boundary_values, _a_macro_dofs);

  // Add entries to global tensor
  A.add(&A_ufc.macro_A[0], a_macro_dofs);
  b.add(&b_ufc.macro_A[0], L_macro_dofs);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble_exterior_facet(GenericMatrix& A, GenericVector& b,
                               UFC& A_ufc, UFC& b_ufc,
                               const Form& a, const Form& L,
                               const Cell& cell, const Facet& facet,
                               Scratch& data,
                               const DirichletBC::Map& boundary_values)
{
  const uint local_facet = cell.index(facet);

  if (A_ufc.form.num_exterior_facet_domains() > 0 )
  {
    ufc::exterior_facet_integral* A_facet_integral = A_ufc.exterior_facet_integrals[0].get();

    A_ufc.update(cell, local_facet);
    A_facet_integral->tabulate_tensor(&A_ufc.A[0], A_ufc.w(), A_ufc.cell, local_facet);
    for (uint i = 0; i < data.Ae.size(); i++)
      data.Ae[i] += A_ufc.A[i];
  }
  if (b_ufc.form.num_exterior_facet_domains() > 0 )
  {
    const ufc::exterior_facet_integral* b_facet_integral = b_ufc.exterior_facet_integrals[0].get();

    b_ufc.update(cell, local_facet);
    b_facet_integral->tabulate_tensor(&b_ufc.A[0], b_ufc.w(), b_ufc.cell, local_facet);
    for (uint i = 0; i < data.be.size(); i++)
      data.be[i] += b_ufc.A[i];
  }

  // If we have local facet 0, assemble cell integral
  if (local_facet == 0)
  {
    if (A_ufc.form.num_cell_domains() > 0 )
    {
      const ufc::cell_integral* A_cell_integral = A_ufc.cell_integrals[0].get();

      A_ufc.update(cell);
      A_cell_integral->tabulate_tensor(&A_ufc.A[0], A_ufc.w(), A_ufc.cell);
      for (uint i = 0; i < data.Ae.size(); i++)
        data.Ae[i] += A_ufc.A[i];
    }

    if (b_ufc.form.num_cell_domains() > 0 )
    {
      const ufc::cell_integral* b_cell_integral = b_ufc.cell_integrals[0].get();

      b_ufc.update(cell);
      b_cell_integral->tabulate_tensor(&b_ufc.A[0], b_ufc.w(), b_ufc.cell);
      for (uint i = 0; i < data.be.size(); i++)
        data.be[i] += b_ufc.A[i];
    }
  }

  // Tabulate dofs
  const std::size_t cell_index = cell.index();
  std::vector<const std::vector<DolfinIndex>* > a_dofs(2);
  std::vector<const std::vector<DolfinIndex>* > L_dofs(1);
  a_dofs[0] = &(a.function_space(0)->dofmap()->cell_dofs(cell_index));
  a_dofs[1] = &(a.function_space(1)->dofmap()->cell_dofs(cell_index));
  L_dofs[0] = &(L.function_space(0)->dofmap()->cell_dofs(cell_index));

  // Modify local matrix/element for Dirichlet boundary conditions
  apply_bc(&data.Ae[0], &data.be[0], boundary_values, a_dofs);

  // Add entries to global tensor
  A.add(&data.Ae[0], a_dofs);
  b.add(&data.be[0], L_dofs);
}
//-----------------------------------------------------------------------------
SystemAssembler::Scratch::Scratch(const Form& a, const Form& L)
{
  uint A_num_entries  = a.function_space(0)->dofmap()->max_cell_dimension();
  A_num_entries      *= a.function_space(1)->dofmap()->max_cell_dimension();
  Ae.resize(A_num_entries);

  be.resize(L.function_space(0)->dofmap()->max_cell_dimension());
}
//-----------------------------------------------------------------------------
SystemAssembler::Scratch::~Scratch()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
inline void SystemAssembler::Scratch::zero_cell()
{
  std::fill(Ae.begin(), Ae.end(), 0.0);
  std::fill(be.begin(), be.end(), 0.0);
}
//-----------------------------------------------------------------------------
