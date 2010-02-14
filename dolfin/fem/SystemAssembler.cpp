// Copyright (C) 2008-2009 Kent-Andre Mardal and Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008-2009.
//
// First added:  2009-06-22
// Last changed: 2009-10-08

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/FunctionSpace.h>
#include "DofMap.h"
#include "FiniteElement.h"
#include "Form.h"
#include "UFC.h"
#include "DirichletBC.h"
#include "AssemblerTools.h"
#include "SystemAssembler.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void SystemAssembler::assemble(GenericMatrix& A,
                                      GenericVector& b,
                                      const Form& a,
                                      const Form& L,
                                      bool reset_sparsitys,
                                      bool add_values)
{
  std::vector<const DirichletBC*> bcs;
  assemble(A, b, a, L, bcs, 0, 0, 0, 0, reset_sparsitys, add_values);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(GenericMatrix& A,
                                          GenericVector& b,
                                          const Form& a,
                                          const Form& L,
                                          const DirichletBC& bc,
                                          bool reset_sparsitys,
                                          bool add_values)
{
  std::vector<const DirichletBC*> bcs;
  bcs.push_back(&bc);
  assemble(A, b, a, L, bcs, 0, 0, 0, 0, reset_sparsitys, add_values);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(GenericMatrix& A,
                                          GenericVector& b,
                                          const Form& a,
                                          const Form& L,
                                          const std::vector<const DirichletBC*>& bcs,
                                          bool reset_sparsitys,
                                          bool add_values)
{
  assemble(A, b, a, L, bcs, 0, 0, 0, 0, reset_sparsitys, add_values);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(GenericMatrix& A,
                                          GenericVector& b,
                                          const Form& a,
                                          const Form& L,
                                          const std::vector<const DirichletBC*>& bcs,
                                          const MeshFunction<uint>* cell_domains,
                                          const MeshFunction<uint>* exterior_facet_domains,
                                          const MeshFunction<uint>* interior_facet_domains,
                                          const GenericVector* x0,
                                          bool reset_sparsitys,
                                          bool add_values)
{
  Timer timer("Assemble system");
  info("Assembling linear system and applying boundary conditions...");

  // FIXME: 1. Need consistency check between a and L
  // FIXME: 2. Some things can be simplified since we know it's a matrix and a vector

  if (cell_domains || exterior_facet_domains || interior_facet_domains)
    error("SystemAssembler does not yet support subdomains.");

  // Check arguments
  AssemblerTools::check(a);
  AssemblerTools::check(L);

  // Check that we have a bilinear and a linear form
  assert(a.rank() == 2);
  assert(L.rank() == 1);

  // Create data structure for local assembly data
  UFC A_ufc(a);
  UFC b_ufc(L);

  // Initialize global tensor
  AssemblerTools::init_global_tensor(A, a, A_ufc, reset_sparsitys, add_values);
  AssemblerTools::init_global_tensor(b, L, b_ufc, reset_sparsitys, add_values);

  // Allocate data
  Scratch data(a, L);

  // Get boundary values (global)
  for(uint i = 0; i < bcs.size(); ++i)
    bcs[i]->get_bc(data.indicators, data.g);

  // Modify boundary values for incremental (typically nonlinear) problems
  if (x0)
  {
    const uint N = a.function_space(1)->dofmap().global_dimension();
    assert(x0->size() == N);
    boost::scoped_array<double> x0_values(double[N]);
    x0->get_local(x0_values.get());
    for (uint i = 0; i < N; i++)
      data.g[i] = x0_values[i] - data.g[i];
  }

  if (A_ufc.form.num_interior_facet_integrals() == 0 && b_ufc.form.num_interior_facet_integrals() == 0)
  {
    // Assemble cell-wise (no interior facet integrals)
    cell_wise_assembly(A, b, a, L, A_ufc, b_ufc, data, cell_domains,
                       exterior_facet_domains);
  }
  else
  {
    not_working_in_parallel("Assembly over interior facets");

    // Assemble facet-wise (including cell assembly)
    facet_wise_assembly(A, b, a, L, A_ufc, b_ufc, data, cell_domains,
                        exterior_facet_domains, interior_facet_domains);
  }

  // Finalise assembly
  A.apply();
  b.apply();
}
//-----------------------------------------------------------------------------
void SystemAssembler::cell_wise_assembly(GenericMatrix& A, GenericVector& b,
                                    const Form& a, const Form& L,
                                    UFC& A_ufc, UFC& b_ufc, Scratch& data,
                                    const MeshFunction<uint>* cell_domains,
                                    const MeshFunction<uint>* exterior_facet_domains)
{
  // FIXME: We can used some std::vectors or array pointers for the A and b
  // related terms to cut down on code repetition.

  const Mesh& mesh = a.mesh();

  // Cell integrals
  ufc::cell_integral* A_integral = A_ufc.cell_integrals[0];
  ufc::cell_integral* b_integral = b_ufc.cell_integrals[0];

  // Exterior facet integrals
  ufc::exterior_facet_integral* A_facet_integral = A_ufc.exterior_facet_integrals[0];
  ufc::exterior_facet_integral* b_facet_integral = b_ufc.exterior_facet_integrals[0];

  Progress p("Assembling system (cell-wise)", mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Update to current cell
    A_ufc.update(*cell);
    b_ufc.update(*cell);

    // Tabulate cell tensor (A)
    A_integral->tabulate_tensor(A_ufc.A.get(), A_ufc.w, A_ufc.cell);
    for (uint i = 0; i < data.A_num_entries; i++)
      data.Ae[i] = A_ufc.A[i];

    // Tabulate cell tensor (b)
    b_integral->tabulate_tensor(b_ufc.A.get(), b_ufc.w, b_ufc.cell);
    for (uint i = 0; i < data.b_num_entries; i++)
      data.be[i] = b_ufc.A[i];

    // FIXME: Can be the assembly over facets be more efficient?
    // Compute exterior facet integral if present
    if (A_ufc.form.num_exterior_facet_integrals() > 0 || b_ufc.form.num_exterior_facet_integrals() > 0)
    {
      const uint D = mesh.topology().dim();
      for (FacetIterator facet(*cell); !facet.end(); ++facet)
      {
        // Assemble if we have an external facet
        if (facet->num_entities(D) != 2)
        {
          const uint local_facet = cell->index(*facet);
          if (A_ufc.form.num_exterior_facet_integrals() > 0)
          {
            A_ufc.update(*cell, local_facet);

            A_facet_integral->tabulate_tensor(A_ufc.A.get(), A_ufc.w, A_ufc.cell, local_facet);
            for (uint i = 0; i < data.A_num_entries; i++)
              data.Ae[i] += A_ufc.A[i];
          }
          if (b_ufc.form.num_exterior_facet_integrals() > 0)
          {
            b_ufc.update(*cell, local_facet);

            b_facet_integral->tabulate_tensor(b_ufc.A.get(), b_ufc.w, b_ufc.cell, local_facet);
            for (uint i = 0; i < data.b_num_entries; i++)
              data.be[i] += b_ufc.A[i];
          }
        }
      }
    }

    // Tabulate dofs for each dimension
    a.function_space(0)->dofmap().tabulate_dofs(A_ufc.dofs[0], A_ufc.cell, cell->index());
    L.function_space(0)->dofmap().tabulate_dofs(b_ufc.dofs[0], b_ufc.cell, cell->index());
    a.function_space(1)->dofmap().tabulate_dofs(A_ufc.dofs[1], A_ufc.cell, cell->index());

    // Modify local matrix/element for Dirichlet boundary conditions
    apply_bc(data.Ae, data.be, data.indicators, data.g, A_ufc.dofs,
             A_ufc.local_dimensions.get());

    // Add entries to global tensor
    A.add(data.Ae, A_ufc.local_dimensions.get(), A_ufc.dofs);
    b.add(data.be, b_ufc.local_dimensions.get(), b_ufc.dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::facet_wise_assembly(GenericMatrix& A, GenericVector& b,
                                    const Form& a, const Form& L,
                                    UFC& A_ufc, UFC& b_ufc, Scratch& data,
                                    const MeshFunction<uint>* cell_domains,
                                    const MeshFunction<uint>* exterior_facet_domains,
                                    const MeshFunction<uint>* interior_facet_domains)
{
  const Mesh& mesh = a.mesh();
  const std::vector<const GenericFunction*> A_coefficients = a.coefficients();
  const std::vector<const GenericFunction*> b_coefficients = L.coefficients();

  Progress p("Assembling system (facet-wise)", mesh.num_facets());
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Interior facet
    if (facet->num_entities(mesh.topology().dim()) == 2)
    {
      // Reset some temp data
      for (uint i = 0; i < A_ufc.macro_local_dimensions[0]*A_ufc.macro_local_dimensions[1]; i++)
        A_ufc.macro_A[i] = 0.0;
      for (uint i = 0; i < b_ufc.macro_local_dimensions[0]; i++)
        b_ufc.macro_A[i] = 0.0;

      // Get cells incident with facet and update UFC objects
      Cell cell0(mesh, facet->entities(mesh.topology().dim())[0]);
      Cell cell1(mesh, facet->entities(mesh.topology().dim())[1]);

      // Get local facet index
      const uint local_facet0 = cell0.index(*facet);
      const uint local_facet1 = cell1.index(*facet);

      A_ufc.update(cell0, local_facet0, cell1, local_facet1);
      b_ufc.update(cell0, local_facet0, cell1, local_facet1);

      // Assemble interior facet and neighbouring cells if needed
      assemble_interior_facet(A, b, A_ufc, b_ufc, a, L, cell0, cell1, *facet, data);
    }

    // Exterior facet
    if ( facet->num_entities(mesh.topology().dim()) != 2 )
    {
      // Reset some temp data
      for (uint i = 0; i < A_ufc.local_dimensions[0]*A_ufc.local_dimensions[1]; i++)
        A_ufc.A[i] = 0.0;
      for (uint i = 0; i < b_ufc.local_dimensions[0]; i++)
        b_ufc.A[i] = 0.0;

      // Get mesh cell to which mesh facet belongs (pick first, there is only one)
      Cell cell(mesh, facet->entities(mesh.topology().dim())[0]);

      // Initialize macro element matrix/vector to zero
      data.zero_cell();

      // Assemble exterior facet and attached cells if needed
      assemble_exterior_facet(A, b, A_ufc, b_ufc, a, L, cell, *facet, data);
    }

    p++;
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::compute_tensor_on_one_interior_facet(const Form& a,
            UFC& ufc, const Cell& cell0, const Cell& cell1, const Facet& facet,
            const MeshFunction<uint>* interior_facet_domains)
{
  const std::vector<const GenericFunction*> coefficients = a.coefficients();

  // Facet integral
  ufc::interior_facet_integral* interior_facet_integral = ufc.interior_facet_integrals[0];

  // Get integral for sub domain (if any)
  if (interior_facet_domains && interior_facet_domains->size() > 0)
  {
    const uint domain = (*interior_facet_domains)[facet];
    if (domain < ufc.form.num_interior_facet_integrals())
      interior_facet_integral = ufc.interior_facet_integrals[domain];
  }

  // Get local index of facet with respect to each cell
  uint local_facet0 = cell0.index(facet);
  uint local_facet1 = cell1.index(facet);

  // Update to current pair of cells
  ufc.update(cell0, local_facet0, cell1, local_facet1);

  interior_facet_integral->tabulate_tensor(ufc.macro_A.get(), ufc.macro_w,
                                           ufc.cell0, ufc.cell1,
                                           local_facet0, local_facet1);
}
//-----------------------------------------------------------------------------
inline void SystemAssembler::apply_bc(double* A, double* b,
                                      const uint* indicators,
                                      const double* g, uint** global_dofs,
                                      const uint* dims)
{
  const uint m = dims[0];
  const uint n = dims[1];

  for (uint i = 0; i < n; i++)
  {
    const uint ii = global_dofs[1][i];
    if (indicators[ii] > 0)
    {
      b[i] = g[ii];
      for (uint k = 0; k < n; k++)
        A[k+i*n] = 0.0;
      for (uint j = 0; j < m; j++)
      {
        b[j] -= A[i+j*n]*g[ii];
        A[i+j*n] = 0.0;
      }
      A[i+i*n] = 1.0;
    }
  }
}
//-----------------------------------------------------------------------------
SystemAssembler::Scratch::Scratch(const Form& a, const Form& L)
  : A_num_entries(1), b_num_entries(1), Ae(0), be(0), indicators(0), g(0)
{
  for (uint i = 0; i < a.rank(); i++)
    A_num_entries *= a.function_space(i)->dofmap().max_local_dimension();
  Ae = new double[A_num_entries];

  for (uint i = 0; i < L.rank(); i++)
    b_num_entries *= L.function_space(i)->dofmap().max_local_dimension();
  be = new double[b_num_entries];

  const uint N = a.function_space(1)->dofmap().global_dimension();
  indicators = new uint[N];
  g = new double[N];
  for (uint i = 0; i < N; i++)
  {
    indicators[i] = 0;
    g[i] = 0.0;
  }
}
//-----------------------------------------------------------------------------
SystemAssembler::Scratch::~Scratch()
{
  delete [] Ae;
  delete [] be;
  delete [] indicators;
  delete [] g;
}
//-----------------------------------------------------------------------------
inline void SystemAssembler::Scratch::zero_cell()
{
  if (Ae)
  {
    for (uint i = 0; i < A_num_entries; i++)
      Ae[i] = 0.0;
  }
  if (be)
  {
    for (uint i = 0; i < b_num_entries; i++)
      be[i] = 0.0;
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble_interior_facet(GenericMatrix& A, GenericVector& b,
                               UFC& A_ufc, UFC& b_ufc,
                               const Form& a,
                               const Form& L,
                               const Cell& cell0, const Cell& cell1, const Facet& facet,
                               const Scratch& data)
{
  // Cell integrals
  ufc::cell_integral* A_cell_integral = A_ufc.cell_integrals[0];
  ufc::cell_integral* b_cell_integral = b_ufc.cell_integrals[0];

  // Compute facet contribution to A
  if (A_ufc.form.num_interior_facet_integrals() > 0)
    compute_tensor_on_one_interior_facet(a, A_ufc, cell0, cell1, facet, 0);

  // Compute facet contribution to
  if (b_ufc.form.num_interior_facet_integrals() > 0)
    compute_tensor_on_one_interior_facet(L, b_ufc, cell0, cell1, facet, 0);

  // Get local facet index
  const uint facet0 = cell0.index(facet);
  const uint facet1 = cell1.index(facet);

  // If we have local facet 0, compute cell contribution
  if (facet0 == 0)
  {
    if (A_ufc.form.num_cell_integrals() > 0)
    {
      A_ufc.update(cell0);

      A_cell_integral->tabulate_tensor(A_ufc.A.get(), A_ufc.w, A_ufc.cell);
      const uint nn = A_ufc.local_dimensions[0];
      const uint mm = A_ufc.local_dimensions[1];
      for (uint i = 0; i < mm; i++)
        for (uint j = 0; j < nn; j++)
          A_ufc.macro_A[2*i*nn+j] += A_ufc.A[i*nn+j];
    }

    if (b_ufc.form.num_cell_integrals() > 0)
    {
      b_ufc.update(cell0);

      b_cell_integral->tabulate_tensor(b_ufc.A.get(), b_ufc.w, b_ufc.cell);
      for (uint i = 0; i < b_ufc.local_dimensions[0]; i++)
        b_ufc.macro_A[i] += b_ufc.A[i];
    }
  }

  // If we have local facet 0, compute and add cell contribution
  if (facet1 == 0)
  {
    if (A_ufc.form.num_cell_integrals() > 0)
    {
      A_ufc.update(cell1);

      A_cell_integral->tabulate_tensor(A_ufc.A.get(), A_ufc.w, A_ufc.cell);
      const uint nn = A_ufc.local_dimensions[0];
      const uint mm = A_ufc.local_dimensions[1];
      for (uint i=0; i < mm; i++)
        for (uint j=0; j < nn; j++)
          A_ufc.macro_A[2*nn*mm + 2*i*nn + nn + j] += A_ufc.A[i*nn+j];
    }

    if (b_ufc.form.num_cell_integrals() > 0)
    {
      b_ufc.update(cell1);

      b_cell_integral->tabulate_tensor(b_ufc.A.get(), b_ufc.w, b_ufc.cell);
      for (uint i=0; i < b_ufc.local_dimensions[0]; i++)
        b_ufc.macro_A[i + b_ufc.local_dimensions[0]] += b_ufc.A[i];
    }
  }

  // Tabulate dofs
  for (uint i = 0; i < A_ufc.form.rank(); i++)
  {
    const uint offset = A_ufc.local_dimensions[i];
    a.function_space(i)->dofmap().tabulate_dofs(A_ufc.macro_dofs[i],
                                               A_ufc.cell0, cell0.index());
    a.function_space(i)->dofmap().tabulate_dofs(A_ufc.macro_dofs[i] + offset,
                                               A_ufc.cell1, cell1.index());
  }
  const uint offset = b_ufc.local_dimensions[0];
  L.function_space(0)->dofmap().tabulate_dofs(b_ufc.macro_dofs[0],
                                             b_ufc.cell0, cell0.index());
  L.function_space(0)->dofmap().tabulate_dofs(b_ufc.macro_dofs[0] + offset,
                                             b_ufc.cell1, cell1.index());

  // Modify local matrix/element for Dirichlet boundary conditions
  apply_bc(A_ufc.macro_A.get(), b_ufc.macro_A.get(), data.indicators, data.g,
           A_ufc.macro_dofs, A_ufc.macro_local_dimensions.get());

  // Add entries to global tensor
  A.add(A_ufc.macro_A.get(), A_ufc.macro_local_dimensions.get(), A_ufc.macro_dofs);
  b.add(b_ufc.macro_A.get(), b_ufc.macro_local_dimensions.get(), b_ufc.macro_dofs);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble_exterior_facet(GenericMatrix& A, GenericVector& b,
                               UFC& A_ufc, UFC& b_ufc,
                               const Form& a,
                               const Form& L,
                               const Cell& cell, const Facet& facet,
                               const Scratch& data)
{
  // Cell integrals
  ufc::cell_integral* A_cell_integral = A_ufc.cell_integrals[0];
  ufc::cell_integral* b_cell_integral = b_ufc.cell_integrals[0];

  // Facet integrals
  ufc::exterior_facet_integral* A_facet_integral = A_ufc.exterior_facet_integrals[0];
  ufc::exterior_facet_integral* b_facet_integral = b_ufc.exterior_facet_integrals[0];

  const uint local_facet = cell.index(facet);

  if (A_ufc.form.num_exterior_facet_integrals() > 0 )
  {
    A_ufc.update(cell, local_facet);

    A_facet_integral->tabulate_tensor(A_ufc.A.get(), A_ufc.w, A_ufc.cell, local_facet);
    for (uint i = 0; i < data.A_num_entries; i++)
      data.Ae[i] += A_ufc.A[i];
  }
  if (b_ufc.form.num_exterior_facet_integrals() > 0 )
  {
    b_ufc.update(cell, local_facet);

    b_facet_integral->tabulate_tensor(b_ufc.A.get(), b_ufc.w, b_ufc.cell, local_facet);
    for (uint i = 0; i < data.b_num_entries; i++)
      data.be[i] += b_ufc.A[i];
  }

  // If we have local facet 0, assemble cell integral
  if (local_facet == 0)
  {
    if (A_ufc.form.num_cell_integrals() > 0 )
    {
      A_ufc.update(cell);
      A_cell_integral->tabulate_tensor(A_ufc.A.get(), A_ufc.w, A_ufc.cell);
      for (uint i = 0; i < data.A_num_entries; i++)
        data.Ae[i] += A_ufc.A[i];
    }

    if (b_ufc.form.num_cell_integrals() > 0 )
    {
      b_ufc.update(cell);
      b_cell_integral->tabulate_tensor(b_ufc.A.get(), b_ufc.w, b_ufc.cell);
      for (uint i = 0; i < data.b_num_entries; i++)
        data.be[i] += b_ufc.A[i];
    }
  }

  // Tabulate dofs
  for (uint i = 0; i < A_ufc.form.rank(); i++)
    a.function_space(i)->dofmap().tabulate_dofs(A_ufc.dofs[i],
                                             A_ufc.cell, cell.index());
  L.function_space(0)->dofmap().tabulate_dofs(b_ufc.dofs[0],
                                             b_ufc.cell, cell.index());

  // Modify local matrix/element for Dirichlet boundary conditions
  apply_bc(data.Ae, data.be, data.indicators, data.g, A_ufc.dofs,
           A_ufc.local_dimensions.get());

  // Add entries to global tensor
  A.add(data.Ae, A_ufc.local_dimensions.get(), A_ufc.dofs);
  b.add(data.be, b_ufc.local_dimensions.get(), b_ufc.dofs);
}
//-----------------------------------------------------------------------------

