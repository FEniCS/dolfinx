// Copyright (C) 2008-2009 Kent-Andre Mardal.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2008-2009.
// Modified by Anders Logg, 2008-2009.
//
// First added:  2009-06-22
// Last changed: 2009-06-22

#include <dolfin/log/dolfin_log.h>
#include <dolfin/common/Timer.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericTensor.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/Scalar.h>
#include <dolfin/la/SparsityPattern.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/MeshData.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/SubDomain.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include "DofMap.h"
#include "Form.h"
#include "UFC.h"
#include "SparsityPatternBuilder.h"
#include "DirichletBC.h"
#include "FiniteElement.h"
#include "Assembler.h"
#include "SystemAssembler.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void SystemAssembler::assemble_system(GenericMatrix& A,
                                      GenericVector& b,
                                      const Form& a,
                                      const Form& L,
                                      bool reset_tensors)
{
  std::vector<const DirichletBC*> bcs;
  assemble_system(A, b, a, L, bcs, 0, 0, 0, 0, reset_tensors);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble_system(GenericMatrix& A,
                                      GenericVector& b,
                                      const Form& a,
                                      const Form& L,
                                      const DirichletBC& bc,
                                      bool reset_tensors)
{
  std::vector<const DirichletBC*> bcs;
  bcs.push_back(&bc);
  assemble_system(A, b, a, L, bcs, 0, 0, 0, 0, reset_tensors);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble_system(GenericMatrix& A,
                                      GenericVector& b,
                                      const Form& a,
                                      const Form& L,
                                      std::vector<const DirichletBC*>& bcs,
                                      bool reset_tensors)
{
  assemble_system(A, b, a, L, bcs, 0, 0, 0, 0, reset_tensors);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble_system(GenericMatrix& A,
                                      GenericVector& b,
                                      const Form& a,
                                      const Form& L,
                                      std::vector<const DirichletBC*>& bcs,
                                      const MeshFunction<uint>* cell_domains,
                                      const MeshFunction<uint>* exterior_facet_domains,
                                      const MeshFunction<uint>* interior_facet_domains,
                                      const GenericVector* x0,
                                      bool reset_tensors)
{
  Timer timer("Assemble system");
  info("Assembling linear system and applying boundary conditions...");

  // FIXME: 1. Need consistency check between a and L
  // FIXME: 2. We assume that we get a bilinear and linear form, need to check this
  // FIXME: 3. Some things can be simplified since we know it's a matrix and a vector

  // Check arguments
  Assembler::check(a);
  Assembler::check(L);

  // Extract mesh and coefficients
  const Mesh& mesh = a.mesh();
  const std::vector<const Function*>& A_coefficients = a.coefficients();
  const std::vector<const Function*>& b_coefficients = L.coefficients();

  // Create data structure for local assembly data
  UFC A_ufc(a);
  UFC b_ufc(L);

  // Initialize global tensor
  Assembler::init_global_tensor(A, a, A_ufc, reset_tensors);
  Assembler::init_global_tensor(b, L, b_ufc, reset_tensors);

  // Pointers to element matrix and vector
  double* Ae = 0;
  double* be = 0;

  // Get boundary values (global)
  const uint N = a.function_space(1).dofmap().global_dimension();
  uint* indicators = new uint[N];
  double* g = new double[N];
  for (uint i = 0; i < N; i++)
  {
    indicators[i] = 0;
    g[i] = 0.0;
  }
  for(uint i = 0; i < bcs.size(); ++i)
    bcs[i]->get_bc(indicators, g);

  // Modify boundary values for incremental (typically nonlinear) problems
  if (x0)
  {
    warning("Symmetric application of Dirichlet boundary conditions for incremental problems is untested.");
    assert( x0->size() == N);
    double* x0_values = new double[N];
    x0->get(x0_values);
    for (uint i = 0; i < N; i++)
      g[i] = x0_values[i] - g[i];
    delete [] x0_values;
  }

  // If there are no interior facet integrals
  if (A_ufc.form.num_interior_facet_integrals() == 0 && b_ufc.form.num_interior_facet_integrals() == 0)
  {
    // Allocate memory for Ae and be
    uint A_num_entries = 1;
    for (uint i = 0; i < a.rank(); i++)
      A_num_entries *= a.function_space(i).dofmap().max_local_dimension();
    Ae = new double[A_num_entries];

    uint b_num_entries = 1;
    for (uint i = 0; i < L.rank(); i++)
      b_num_entries *= L.function_space(i).dofmap().max_local_dimension();
    be = new double[b_num_entries];

    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      for (uint i = 0; i < A_num_entries; i++)
        Ae[i] = 0.0;
      for (uint i = 0; i < b_num_entries; i++)
        be[i] = 0.0;

      // Update to current cell
      A_ufc.update(*cell);

      // Interpolate coefficients on cell
      for (uint i = 0; i < A_coefficients.size(); i++)
        A_coefficients[i]->interpolate(A_ufc.w[i], A_ufc.cell, cell->index());

      // Tabulate dofs for each dimension
      for (uint i = 0; i < A_ufc.form.rank(); i++)
        a.function_space(i).dofmap().tabulate_dofs(A_ufc.dofs[i], A_ufc.cell, cell->index());

      // Update to current cell
      b_ufc.update(*cell);

      // Interpolate coefficients on cell
      for (uint i = 0; i < b_coefficients.size(); i++)
        b_coefficients[i]->interpolate(b_ufc.w[i], b_ufc.cell, cell->index());

      // Tabulate dofs for each dimension
      for (uint i = 0; i < b_ufc.form.rank(); i++)
        L.function_space(i).dofmap().tabulate_dofs(b_ufc.dofs[i], b_ufc.cell, cell->index());

      // Compute cell integral for A
      if (A_ufc.form.num_cell_integrals() > 0)
      {
        ufc::cell_integral* A_cell_integral =  A_ufc.cell_integrals[0];
        if (cell_domains && cell_domains->size() > 0)
        {
          const uint cell_domain = (*cell_domains)(*cell);
          if (cell_domain < A_ufc.form.num_cell_integrals())
            A_cell_integral = A_ufc.cell_integrals[cell_domain];
          else
            continue;
        }
        // Tabulate cell tensor
        A_cell_integral->tabulate_tensor(A_ufc.A, A_ufc.w, A_ufc.cell);
        for (uint i=0; i<A_num_entries; i++)
          Ae[i] += A_ufc.A[i];
      }

      // Compute cell integral for b
      if (b_ufc.form.num_cell_integrals() > 0)
      {
        ufc::cell_integral* b_cell_integral = b_ufc.cell_integrals[0];
        if (cell_domains && cell_domains->size() > 0)
        {
          const uint cell_domain = (*cell_domains)(*cell);
          if (cell_domain < b_ufc.form.num_cell_integrals())
            b_cell_integral = b_ufc.cell_integrals[cell_domain];
          else
            continue;
        }
        // Tabulate cell tensor
        b_cell_integral->tabulate_tensor(b_ufc.A, b_ufc.w, b_ufc.cell);
        for (uint i=0; i < b_num_entries; i++)
          be[i] += b_ufc.A[i];
      }

      // Compute exterior facet integral
      if (A_ufc.form.num_exterior_facet_integrals() > 0 || b_ufc.form.num_exterior_facet_integrals() > 0)
      {
        const uint D = mesh.topology().dim();

        if (A_ufc.form.num_exterior_facet_integrals() > 0)
        {
          for (FacetIterator facet(*cell); !facet.end(); ++facet)
          {
            if (facet->num_entities(D) != 2)
            {
              ufc::exterior_facet_integral* A_integral = A_ufc.exterior_facet_integrals[0];
              if (exterior_facet_domains && exterior_facet_domains->size() > 0)
              {
                const uint exterior_facet_domain= (*exterior_facet_domains)(*facet);
                if (exterior_facet_domain < A_ufc.form.num_exterior_facet_integrals())
                  A_integral = A_ufc.exterior_facet_integrals[exterior_facet_domain];
                else
                  continue;
              }
              const uint local_facet = cell->index(*facet);
              for (uint i = 0; i < A_coefficients.size(); i++)
                A_coefficients[i]->interpolate(A_ufc.w[i], A_ufc.cell, cell->index(), local_facet);

              A_integral->tabulate_tensor(A_ufc.A, A_ufc.w, A_ufc.cell, local_facet);
              for (uint i=0; i<A_num_entries; i++)
                Ae[i] += A_ufc.A[i];
            }
          }
        }

        if (b_ufc.form.num_exterior_facet_integrals() > 0)
        {
          for (FacetIterator facet(*cell); !facet.end(); ++facet)
          {
            if (facet->num_entities(D) != 2)
            {
              ufc::exterior_facet_integral* b_integral = b_ufc.exterior_facet_integrals[0];
              if (exterior_facet_domains && exterior_facet_domains->size() > 0)
              {
                const uint exterior_facet_domain= (*exterior_facet_domains)(*facet);
                if (exterior_facet_domain < b_ufc.form.num_exterior_facet_integrals())
                  b_integral = b_ufc.exterior_facet_integrals[exterior_facet_domain];
                else
                  continue;
              }
              const uint local_facet = cell->index(*facet);
              for (uint i = 0; i < b_coefficients.size(); i++)
                b_coefficients[i]->interpolate(b_ufc.w[i], b_ufc.cell, cell->index(), local_facet);

              b_integral->tabulate_tensor(b_ufc.A, b_ufc.w, b_ufc.cell, local_facet);
              for (uint i=0; i < b_num_entries; i++)
                be[i] += b_ufc.A[i];
            }
          }
        }
      }

      // Enforce Dirichlet boundary conditions
      /*
      const uint m = A_ufc.local_dimensions[0];
      const uint n = A_ufc.local_dimensions[1];
      for (uint i=0; i<n; i++)
      {
        uint ii = A_ufc.dofs[1][i];
        if (indicators[ii] > 0)
        {
          be[i] = g[ii];
          for (uint k=0; k<n; k++)
            Ae[k+i*n] = 0.0;
          for (uint j=0; j<m; j++)
          {
            be[j] -= Ae[i+j*n]*g[ii];
            Ae[i+j*n] = 0.0;
          }
          Ae[i+i*n] = 1.0;
        }
      }
      */
      // Add entries to global tensor
      A.add(Ae, A_ufc.local_dimensions, A_ufc.dofs);
      b.add(be, b_ufc.local_dimensions, b_ufc.dofs);
    }
  }

  if (A_ufc.form.num_interior_facet_integrals() > 0 || b_ufc.form.num_interior_facet_integrals() > 0)
  {
    // Create temporary storage for Ae, Ae_macro
    uint A_num_entries = 1;
    for (uint i = 0; i < a.rank(); i++)
      A_num_entries *= a.function_space(i).dofmap().max_local_dimension();
    uint A_macro_num_entries = 4*A_num_entries;
    Ae = new double[A_num_entries];
    double* Ae_macro = new double[A_macro_num_entries];

    // Create temporay storage for be, be_macro
    uint b_num_entries = 1;
    for (uint i = 0; i < L.rank(); i++)
      b_num_entries *= L.function_space(i).dofmap().max_local_dimension();
    uint b_macro_num_entries = b_num_entries*2;
    be = new double[b_num_entries];
    double* be_macro = new double[b_macro_num_entries];

    for (FacetIterator facet(mesh); !facet.end(); ++facet)
    {
      // Check if we have an interior facet
      if ( facet->num_entities(mesh.topology().dim()) == 2 )
      {
        for (uint i = 0; i < A_macro_num_entries; i++)
          Ae_macro[i] = 0.0;
        for (uint i = 0; i < b_macro_num_entries; i++)
          be_macro[i] = 0.0;

        // Get cells incident with facet
        Cell cell0(mesh, facet->entities(mesh.topology().dim())[0]);
        Cell cell1(mesh, facet->entities(mesh.topology().dim())[1]);

        // Update to current pair of cells
        A_ufc.update(cell0, cell1);
        b_ufc.update(cell0, cell1);

        // Get local index of facet with respect to each cell
        uint facet0 = cell0.index(*facet);
        uint facet1 = cell1.index(*facet);

        // Tabulate dofs for each dimension on macro element
        for (uint i = 0; i < A_ufc.form.rank(); i++)
        {
          const uint offset = A_ufc.local_dimensions[i];
          a.function_space(i).dofmap().tabulate_dofs(A_ufc.macro_dofs[i],
                                                     A_ufc.cell0, cell0.index());
          a.function_space(i).dofmap().tabulate_dofs(A_ufc.macro_dofs[i] + offset,
                                                     A_ufc.cell1, cell1.index());
        }

        // Tabulate dofs for each dimension on macro element
        for (uint i = 0; i < b_ufc.form.rank(); i++)
        {
          const uint offset = b_ufc.local_dimensions[i];
          L.function_space(i).dofmap().tabulate_dofs(b_ufc.macro_dofs[i],
                                                     b_ufc.cell0, cell0.index());
          L.function_space(i).dofmap().tabulate_dofs(b_ufc.macro_dofs[i] + offset,
                                                     b_ufc.cell1, cell1.index());
        }

        if ( A_ufc.form.num_interior_facet_integrals() )
        {
          ufc::interior_facet_integral* interior_facet_integral = A_ufc.interior_facet_integrals[0];

          // Get integral for sub domain (if any)
          if (interior_facet_domains && interior_facet_domains->size() > 0)
          {
            const uint domain = (*interior_facet_domains)(*facet);
            if (domain < A_ufc.form.num_interior_facet_integrals())
              interior_facet_integral = A_ufc.interior_facet_integrals[domain];
            else
              continue;
          }


          // Interpolate coefficients on cell
          for (uint i = 0; i < A_coefficients.size(); i++)
          {
            const uint offset = A_ufc.coefficient_elements[i]->space_dimension();
            A_coefficients[i]->interpolate(A_ufc.macro_w[i],          A_ufc.cell0, cell0.index(), facet0);
            A_coefficients[i]->interpolate(A_ufc.macro_w[i] + offset, A_ufc.cell1, cell1.index(), facet1);
          }


          // Get integral for sub domain (if any)
          if (interior_facet_domains && interior_facet_domains->size() > 0)
          {
            const uint domain = (*interior_facet_domains)(*facet);
            if (domain < A_ufc.form.num_interior_facet_integrals())
              interior_facet_integral = A_ufc.interior_facet_integrals[domain];
            else
              continue;
          }

          // Tabulate interior facet tensor on macro element
          interior_facet_integral->tabulate_tensor(A_ufc.macro_A, A_ufc.macro_w,
                                      A_ufc.cell0, A_ufc.cell1, facet0, facet1);
          for (uint i=0; i<A_macro_num_entries; i++)
            Ae_macro[i] += A_ufc.macro_A[i];
        }

        if ( b_ufc.form.num_interior_facet_integrals() > 0 )
        {
          ufc::interior_facet_integral* interior_facet_integral = b_ufc.interior_facet_integrals[0];

          b_ufc.update(cell0, cell1);

          // Interpolate coefficients on cell
          for (uint i = 0; i < b_coefficients.size(); i++)
          {
            const uint offset = b_ufc.coefficient_elements[i]->space_dimension();
            b_coefficients[i]->interpolate(b_ufc.macro_w[i],          b_ufc.cell0, cell0.index(), facet0);
            b_coefficients[i]->interpolate(b_ufc.macro_w[i] + offset, b_ufc.cell1, cell1.index(), facet1);
          }

          // Get integral for sub domain (if any)
          if (interior_facet_domains && interior_facet_domains->size() > 0)
          {
            const uint domain = (*interior_facet_domains)(*facet);
            if (domain < b_ufc.form.num_interior_facet_integrals())
              interior_facet_integral = b_ufc.interior_facet_integrals[domain];
            else
              continue;
          }

          interior_facet_integral->tabulate_tensor(b_ufc.macro_A, b_ufc.macro_w, b_ufc.cell0,
                                                   b_ufc.cell1, facet0, facet1);

          for (uint i=0; i<b_macro_num_entries; i++)
            be_macro[i] += b_ufc.macro_A[i];
        }

        if (facet0 == 0)
        {
          if (A_ufc.form.num_cell_integrals() > 0)
          {
            A_ufc.update(cell0);

            // Interpolate coefficients on cell
            for (uint i = 0; i < A_coefficients.size(); i++)
              A_coefficients[i]->interpolate(A_ufc.w[i], A_ufc.cell, cell0.index());

            // Tabulate dofs for each dimension
            for (uint i = 0; i < A_ufc.form.rank(); i++)
              a.function_space(i).dofmap().tabulate_dofs(A_ufc.dofs[i], A_ufc.cell, cell0.index());

            ufc::cell_integral* A_cell_integral =  A_ufc.cell_integrals[0];

            if (cell_domains && cell_domains->size() > 0)
            {
              const uint cell_domain = (*cell_domains)(cell0);
              if (cell_domain < A_ufc.form.num_cell_integrals())
                A_cell_integral = A_ufc.cell_integrals[cell_domain];
              else
                continue;
            }
            // Tabulate cell tensor
            A_cell_integral->tabulate_tensor(A_ufc.A, A_ufc.w, A_ufc.cell0);

            uint nn = A_ufc.local_dimensions[0];
            uint mm = A_ufc.local_dimensions[1];
            for (uint i=0; i<mm; i++)
              for (uint j=0; j<nn; j++)
                Ae_macro[2*i*nn+j] += A_ufc.A[i*nn+j];
          }

          if (b_ufc.form.num_cell_integrals() > 0)
          {
            // Update to cell0
            b_ufc.update(cell0);

            // Interpolate coefficients on cell
            for (uint i = 0; i < b_coefficients.size(); i++)
              b_coefficients[i]->interpolate(b_ufc.w[i], b_ufc.cell, cell0.index());

            // Tabulate dofs for each dimension
            for (uint i = 0; i < b_ufc.form.rank(); i++)
              L.function_space(i).dofmap().tabulate_dofs(b_ufc.dofs[i], b_ufc.cell, cell0.index());

            ufc::cell_integral* b_cell_integral =  b_ufc.cell_integrals[0];

            if (cell_domains && cell_domains->size() > 0)
            {
              const uint cell_domain = (*cell_domains)(cell0);
              if (cell_domain < b_ufc.form.num_cell_integrals())
                b_cell_integral = b_ufc.cell_integrals[cell_domain];
              else
                continue;
            }
            // Tabulate cell tensor
            b_cell_integral->tabulate_tensor(b_ufc.A, b_ufc.w, b_ufc.cell0);
            for (uint i=0; i<b_num_entries; i++)
              be_macro[i] += b_ufc.A[i];
          }
        }

        if (facet1 == 0)
        {
          if (A_ufc.form.num_cell_integrals() > 0)
          {
            A_ufc.update(cell1);

            // Interpolate coefficients on cell
            for (uint i = 0; i < A_coefficients.size(); i++)
              A_coefficients[i]->interpolate(A_ufc.w[i], A_ufc.cell, cell1.index());

            // Tabulate dofs for each dimension
            for (uint i = 0; i < A_ufc.form.rank(); i++)
              a.function_space(i).dofmap().tabulate_dofs(A_ufc.dofs[i], A_ufc.cell, cell1.index());

            ufc::cell_integral* A_cell_integral =  A_ufc.cell_integrals[0];

            if (cell_domains && cell_domains->size() > 0)
            {
              const uint cell_domain = (*cell_domains)(cell1);
              if (cell_domain < A_ufc.form.num_cell_integrals())
                A_cell_integral = A_ufc.cell_integrals[cell_domain];
              else
                continue;
            }

            // Tabulate cell tensor
            A_cell_integral->tabulate_tensor(A_ufc.A, A_ufc.w, A_ufc.cell1);
            uint nn = A_ufc.local_dimensions[0];
            uint mm = A_ufc.local_dimensions[1];
            for (uint i=0; i<mm; i++)
              for (uint j=0; j<nn; j++)
                Ae_macro[2*nn*mm + 2*i*nn + nn + j] += A_ufc.A[i*nn+j];
          }

          if (b_ufc.form.num_cell_integrals() > 0)
          {
            b_ufc.update(cell1);

            // Interpolate coefficients on cell
            for (uint i = 0; i < b_coefficients.size(); i++)
              b_coefficients[i]->interpolate(b_ufc.w[i], b_ufc.cell, cell1.index());

            // Tabulate dofs for each dimension
            for (uint i = 0; i < b_ufc.form.rank(); i++)
              L.function_space(i).dofmap().tabulate_dofs(b_ufc.dofs[i], b_ufc.cell, cell1.index());

            ufc::cell_integral* b_cell_integral =  b_ufc.cell_integrals[0];

            if (cell_domains && cell_domains->size() > 0)
            {
              const uint cell_domain = (*cell_domains)(cell1);
              if (cell_domain < b_ufc.form.num_cell_integrals())
                b_cell_integral = b_ufc.cell_integrals[cell_domain];
              else
                continue;
            }
            // Tabulate cell tensor
            b_cell_integral->tabulate_tensor(b_ufc.A, b_ufc.w, b_ufc.cell1);
            for (uint i=0; i<b_num_entries; i++)
              be_macro[b_num_entries + i] += b_ufc.A[i];
          }
        }
        // enforce BC  ---------------------------------------

        /*
        const uint m = A_ufc.macro_local_dimensions[0];
        const uint n = A_ufc.macro_local_dimensions[1];
        for (uint i=0; i<n; i++)
        {
          uint ii = A_ufc.macro_dofs[1][i];
          if (indicators[ii] > 0)
          {
            be_macro[i] = g[ii];
            for (uint k=0; k<n; k++)
              Ae_macro[k+i*n] = 0.0;
            for (uint j=0; j<m; j++)
            {
              be_macro[j] -= Ae_macro[i+j*n]*g[ii];
              Ae_macro[i+j*n] = 0.0;
            }
            Ae_macro[i+i*n] = 1.0;
          }
        }
        */
        // enforce BC done  ------------------------------------------

        // Add entries to global tensor
        A.add(Ae_macro, A_ufc.macro_local_dimensions, A_ufc.macro_dofs);
        b.add(be_macro, b_ufc.macro_local_dimensions, b_ufc.macro_dofs);
      }

      // Check if we have an exterior facet
      if ( facet->num_entities(mesh.topology().dim()) != 2 )
      {
        // set element matrix and vector to zero
        for (uint i=0; i<A_num_entries; i++)
          Ae[i] = 0.0;
        for (uint i=0; i<b_num_entries; i++)
          be[i] = 0.0;

        // Get mesh cell to which mesh facet belongs (pick first, there is only one)
        Cell cell(mesh, facet->entities(mesh.topology().dim())[0]);

        // Get local index of facet with respect to the cell
        const uint local_facet = cell.index(*facet);

        // Update to current cell
        A_ufc.update(cell);
        b_ufc.update(cell);

        // Interpolate coefficients on cell
        for (uint i = 0; i < A_coefficients.size(); i++)
          A_coefficients[i]->interpolate(A_ufc.w[i], A_ufc.cell, cell.index(), local_facet);

        // Interpolate coefficients on cell
        for (uint i = 0; i < b_coefficients.size(); i++)
          b_coefficients[i]->interpolate(b_ufc.w[i], b_ufc.cell, cell.index(), local_facet);

        // Tabulate dofs for each dimension
        for (uint i = 0; i < A_ufc.form.rank(); i++)
          a.function_space(i).dofmap().tabulate_dofs(A_ufc.dofs[i], A_ufc.cell, cell.index());

        // Tabulate dofs for each dimension
        for (uint i = 0; i < b_ufc.form.rank(); i++)
          L.function_space(i).dofmap().tabulate_dofs(b_ufc.dofs[i], b_ufc.cell, cell.index());

        if (local_facet == 0)
        {
          // compute cell integrals ---------------------------------

          // compute cell for A integral ----------------------------
          if (A_ufc.form.num_cell_integrals() > 0)
          {
            ufc::cell_integral* A_cell_integral =  A_ufc.cell_integrals[0];
            if (cell_domains && cell_domains->size() > 0)
            {
              const uint cell_domain = (*cell_domains)(cell);
              if (cell_domain < A_ufc.form.num_cell_integrals())
                A_cell_integral = A_ufc.cell_integrals[cell_domain];
              else
                continue;
            }
            // Tabulate cell tensor
            A_cell_integral->tabulate_tensor(A_ufc.A, A_ufc.w, A_ufc.cell);
            for (uint i=0; i<A_num_entries; i++)
              Ae[i] += A_ufc.A[i];
          }
          // compute cell A integral done ----------------------------

          // compute cell b integral ----------------------------
          if (b_ufc.form.num_cell_integrals() > 0)
          {
            ufc::cell_integral* b_cell_integral = b_ufc.cell_integrals[0];
            if (cell_domains && cell_domains->size() > 0)
            {
              const uint cell_domain = (*cell_domains)(cell);
              if (cell_domain < b_ufc.form.num_cell_integrals())
                b_cell_integral = b_ufc.cell_integrals[cell_domain];
              else
                continue;
            }
            // Tabulate cell tensor
            b_cell_integral->tabulate_tensor(b_ufc.A, b_ufc.w, b_ufc.cell);
            for (uint i=0; i<b_num_entries; i++)
              be[i] += b_ufc.A[i];
          }
          // compute cell b integral done ----------------------------

          // compute cell integral done ------------------------------
        }

        // compute exterior facet integral -------------------------

        if (A_ufc.form.num_exterior_facet_integrals() > 0 )
        {
          const uint D = mesh.topology().dim();
          if (facet->num_entities(D) != 2)
          {
            ufc::exterior_facet_integral* A_integral = A_ufc.exterior_facet_integrals[0];

            if (exterior_facet_domains && exterior_facet_domains->size() > 0)
            {
              const uint exterior_facet_domain= (*exterior_facet_domains)(*facet);
              if (exterior_facet_domain < A_ufc.form.num_exterior_facet_integrals())
                A_integral = A_ufc.exterior_facet_integrals[exterior_facet_domain];
              else
                continue;
            }
            A_integral->tabulate_tensor(A_ufc.A, A_ufc.w, A_ufc.cell, local_facet);
            for (uint i=0; i<A_num_entries; i++)
              Ae[i] += A_ufc.A[i];
          }
        }

        if (b_ufc.form.num_exterior_facet_integrals() > 0)
        {
          const uint D = mesh.topology().dim();
          if (facet->num_entities(D) != 2)
          {
            ufc::exterior_facet_integral* b_integral = b_ufc.exterior_facet_integrals[0];
            if (exterior_facet_domains && exterior_facet_domains->size() > 0)
            {
              const uint exterior_facet_domain= (*exterior_facet_domains)(*facet);
              if (exterior_facet_domain < b_ufc.form.num_exterior_facet_integrals())
                b_integral = b_ufc.exterior_facet_integrals[exterior_facet_domain];
              else
                continue;
            }
            const uint local_facet = cell.index(*facet);
            b_integral->tabulate_tensor(b_ufc.A, b_ufc.w, b_ufc.cell, local_facet);
            for (uint i=0; i<b_num_entries; i++)
              be[i] += b_ufc.A[i];
          }
        }
        // enforce BC  ---------------------------------------

        /*
        uint m = A_ufc.local_dimensions[0];
        uint n = A_ufc.local_dimensions[1];
        for (uint i=0; i<n; i++)
        {
          uint ii = A_ufc.dofs[1][i];
          if (indicators[ii] > 0)
          {
            be[i] = g[ii];
            for (uint k=0; k<n; k++)
              Ae[k+i*n] = 0.0;
            for (uint j=0; j<m; j++)
            {
              be[j] -= Ae[i+j*n]*g[ii];
              Ae[i+j*n] = 0.0;
            }
            Ae[i+i*n] = 1.0;
          }
        }
        */

        // enforce BC done  ------------------------------------------

        // Add entries to global tensor
        //A.add(Ae, A_ufc.local_dimensions, A_ufc.dofs);
        //b.add(be, b_ufc.local_dimensions, b_ufc.dofs);
      }
    }
    delete [] Ae_macro;
    delete [] be_macro;
  }

  // -- Finalize tensors
  A.apply();
  b.apply();

  delete [] Ae;
  delete [] be;
  delete [] g;
  delete [] indicators;
}
//-----------------------------------------------------------------------------
void SystemAssembler::compute_tensor_on_one_cell(const Form& a,
                                                 UFC& ufc, 
                                                 const Cell& cell, 
                                                 const std::vector<const Function*>& coefficients, 
                                                 const MeshFunction<uint>* cell_domains) 
{
    // Cell integral
    ufc::cell_integral* integral = ufc.cell_integrals[0];

    // Get integral for sub domain (if any)
    if (cell_domains && cell_domains->size() > 0)
    {
      const uint domain = (*cell_domains)(cell);
      if (domain < ufc.form.num_cell_integrals())
        integral = ufc.cell_integrals[domain];
    }

    // Update to current cell
    ufc.update(cell);

    // Interpolate coefficients on cell
    for (uint i = 0; i < coefficients.size(); i++)
      coefficients[i]->interpolate(ufc.w[i], ufc.cell, cell.index());

    // Tabulate dofs for each dimension
    for (uint i = 0; i < ufc.form.rank(); i++)
      a.function_space(i).dofmap().tabulate_dofs(ufc.dofs[i], ufc.cell, cell.index());

    // Tabulate cell tensor
    integral->tabulate_tensor(ufc.A, ufc.w, ufc.cell);
}
//-----------------------------------------------------------------------------
void SystemAssembler::compute_tensor_on_one_exterior_facet(const Form& a,
                                                           UFC& ufc, 
                                                           const Cell& cell, 
                                                           const Facet& facet,
                                                           const std::vector<const Function*>& coefficients, 
                                                           const MeshFunction<uint>* exterior_facet_domains) 
{
  // Get facet integral
  ufc::exterior_facet_integral* integral = ufc.exterior_facet_integrals[0];; 

  // Get integral for sub domain (if any)
  if (exterior_facet_domains && exterior_facet_domains->size() > 0)
  {
    const uint exterior_facet_domain= (*exterior_facet_domains)(facet);
    if (exterior_facet_domain < ufc.form.num_exterior_facet_integrals())
      integral = ufc.exterior_facet_integrals[exterior_facet_domain];
  }

  // Update to current cell
  ufc.update(cell);

  // Tabulate dofs for each dimension
  for (uint i = 0; i < ufc.form.rank(); i++)
    a.function_space(i).dofmap().tabulate_dofs(ufc.dofs[i], ufc.cell, cell.index());

  // Interpolate coefficients on cell
  const uint local_facet = cell.index(facet);
  for (uint i = 0; i < coefficients.size(); i++)
    coefficients[i]->interpolate(ufc.w[i], ufc.cell, cell.index(), local_facet);

  integral->tabulate_tensor(ufc.A, ufc.w, ufc.cell, local_facet);
}
//-----------------------------------------------------------------------------
void SystemAssembler::compute_tensor_on_one_interior_facet(const Form& a, 
            UFC& ufc, const Cell& cell0, const Cell& cell1, const Facet& facet,  
            const std::vector<const Function*>& coefficients, 
            const MeshFunction<uint>* interior_facet_domains)
{
  // Facet integral
  ufc::interior_facet_integral* interior_facet_integral = ufc.interior_facet_integrals[0];

  // Get integral for sub domain (if any)
  if (interior_facet_domains && interior_facet_domains->size() > 0)
  {
    const uint domain = (*interior_facet_domains)(facet);
    if (domain < ufc.form.num_interior_facet_integrals())
      interior_facet_integral = ufc.interior_facet_integrals[domain];
  }

  // Update to current pair of cells
  ufc.update(cell0, cell1);

  // Get local index of facet with respect to each cell
  uint facet0 = cell0.index(facet);
  uint facet1 = cell1.index(facet);

  // Tabulate dofs for each dimension on macro element
  for (uint i = 0; i < ufc.form.rank(); i++)
  {
    const uint offset = ufc.local_dimensions[i];
    a.function_space(i).dofmap().tabulate_dofs(ufc.macro_dofs[i],
                                               ufc.cell0, cell0.index());
    a.function_space(i).dofmap().tabulate_dofs(ufc.macro_dofs[i] + offset,
                                               ufc.cell1, cell1.index());
  }

  // Interpolate coefficients on cell
  for (uint i = 0; i < coefficients.size(); i++)
  {
    const uint offset = ufc.coefficient_elements[i]->space_dimension();
    coefficients[i]->interpolate(ufc.macro_w[i],          ufc.cell0, cell0.index(), facet0);
    coefficients[i]->interpolate(ufc.macro_w[i] + offset, ufc.cell1, cell1.index(), facet1);
  }

  interior_facet_integral->tabulate_tensor(ufc.macro_A, ufc.macro_w, 
                                           ufc.cell0, ufc.cell1, 
                                           facet0, facet1);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble_system_new(GenericMatrix& A,
                                      GenericVector& b,
                                      const Form& a,
                                      const Form& L,
                                      bool reset_tensors)
{
  std::vector<const DirichletBC*> bcs;
  assemble_system_new(A, b, a, L, bcs, 0, 0, 0, 0, reset_tensors);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble_system_new(GenericMatrix& A,
                                          GenericVector& b,
                                          const Form& a,
                                          const Form& L,
                                          const DirichletBC& bc,
                                          bool reset_tensors)
{
  std::vector<const DirichletBC*> bcs;
  bcs.push_back(&bc);
  assemble_system_new(A, b, a, L, bcs, 0, 0, 0, 0, reset_tensors);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble_system_new(GenericMatrix& A,
                                          GenericVector& b,
                                          const Form& a,
                                          const Form& L,
                                          std::vector<const DirichletBC*>& bcs,
                                          bool reset_tensors)
{
  assemble_system_new(A, b, a, L, bcs, 0, 0, 0, 0, reset_tensors);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble_system_new(GenericMatrix& A,
                                          GenericVector& b,
                                          const Form& a,
                                          const Form& L,
                                          std::vector<const DirichletBC*>& bcs,
                                          const MeshFunction<uint>* cell_domains,
                                          const MeshFunction<uint>* exterior_facet_domains,
                                          const MeshFunction<uint>* interior_facet_domains,
                                          const GenericVector* x0,
                                          bool reset_tensors)
{
  Timer timer("Assemble system");
  info("Assembling linear system and applying boundary conditions...");

  // FIXME: 1. Need consistency check between a and L
  // FIXME: 2. Some things can be simplified since we know it's a matrix and a vector

  // Check arguments
  Assembler::check(a);
  Assembler::check(L);

  // Check that we have a bilinear and a linear form
  assert(a.rank() == 2);
  assert(L.rank() == 1);

  // Create data structure for local assembly data
  UFC A_ufc(a);
  UFC b_ufc(L);

  // Initialize global tensor
  Assembler::init_global_tensor(A, a, A_ufc, reset_tensors);
  Assembler::init_global_tensor(b, L, b_ufc, reset_tensors);

  // Allocate data
  Scratch data(a, L);

  // Get boundary values (global) 
  for(uint i = 0; i < bcs.size(); ++i)
    bcs[i]->get_bc(data.indicators, data.g);

  // Modify boundary values for incremental (typically nonlinear) problems
  if (x0)
  {
    const uint N = a.function_space(1).dofmap().global_dimension();  
    assert(x0->size() == N);
    double* x0_values = new double[N];
    x0->get(x0_values);
    for (uint i = 0; i < N; i++)
      data.g[i] = x0_values[i] - data.g[i];
    delete [] x0_values;
  }

  if (A_ufc.form.num_interior_facet_integrals() == 0 && b_ufc.form.num_interior_facet_integrals() == 0) 
  {
    // Assmeble cell-wise (no interior facet integrals)
    cell_assembly(A, b, a, L, A_ufc, b_ufc, data, cell_domains, 
                  exterior_facet_domains, interior_facet_domains);
  } 
  else
  { 
    // Assmeble facet-wise (including cell assembly)
    facet_assembly(A, b, a, L, A_ufc, b_ufc, data, cell_domains, 
                  exterior_facet_domains, interior_facet_domains);
  }

  // Finalise assembly
  A.apply();
  b.apply();
}
//-----------------------------------------------------------------------------
void SystemAssembler::cell_assembly(GenericMatrix& A, GenericVector& b,
                                    const Form& a, const Form& L, 
                                    UFC& A_ufc, UFC& b_ufc, Scratch& data, 
                                    const MeshFunction<uint>* cell_domains,
                                    const MeshFunction<uint>* exterior_facet_domains,
                                    const MeshFunction<uint>* interior_facet_domains)
{
  const Mesh& mesh = a.mesh();
  const std::vector<const Function*>& A_coefficients = a.coefficients();
  const std::vector<const Function*>& b_coefficients = L.coefficients();

  for (CellIterator cell(mesh); !cell.end(); ++cell) 
  {
    // Compute element matrix and vector on one cell
    compute_tensor_on_one_cell(a, A_ufc, *cell, A_coefficients, cell_domains); 
    compute_tensor_on_one_cell(L, b_ufc, *cell, b_coefficients, cell_domains); 

    // Compute exterior facet integral
    if (A_ufc.form.num_exterior_facet_integrals() > 0 || b_ufc.form.num_exterior_facet_integrals() > 0) 
    {
      const uint D = mesh.topology().dim(); 
      if (A_ufc.form.num_exterior_facet_integrals() > 0) 
      {
        for (FacetIterator facet(*cell); !facet.end(); ++facet)
        {
          if (facet->num_entities(D) != 2) 
          {
            compute_tensor_on_one_exterior_facet(a, A_ufc, *cell, *facet, A_coefficients, exterior_facet_domains);
            compute_tensor_on_one_exterior_facet(L, b_ufc, *cell, *facet, b_coefficients, exterior_facet_domains);
          }
        }
      }

      // Modify local matrix/element for Dirichlet boundary conditions
      apply_bc(A_ufc.A, b_ufc.A, data.indicators, data.g, A_ufc.dofs, 
               A_ufc.local_dimensions);

      // Add entries to global tensor
      A.add(A_ufc.A, A_ufc.local_dimensions, A_ufc.dofs);
      b.add(b_ufc.A, b_ufc.local_dimensions, b_ufc.dofs);
    }
  }
} 
//-----------------------------------------------------------------------------
void SystemAssembler::facet_assembly(GenericMatrix& A, GenericVector& b,
                                    const Form& a, const Form& L, 
                                    UFC& A_ufc, UFC& b_ufc, Scratch& data, 
                                    const MeshFunction<uint>* cell_domains,
                                    const MeshFunction<uint>* exterior_facet_domains,
                                    const MeshFunction<uint>* interior_facet_domains)
{
  const Mesh& mesh = a.mesh();
  const std::vector<const Function*>& A_coefficients = a.coefficients();
  const std::vector<const Function*>& b_coefficients = L.coefficients();

  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Interior facet
    if ( facet->num_entities(mesh.topology().dim()) == 2 )
    {
      // Initialize macro element matrix/vector to zero
      data.init_macro();

      // Get cells incident with facet and update UFC objects
      Cell cell0(mesh, facet->entities(mesh.topology().dim())[0]);
      Cell cell1(mesh, facet->entities(mesh.topology().dim())[1]);
      A_ufc.update(cell0, cell1);
      b_ufc.update(cell0, cell1);

      if (A_ufc.form.num_interior_facet_integrals() > 0)
      {
        compute_tensor_on_one_interior_facet(a, A_ufc, cell0, cell1, *facet, 
                                             A_coefficients, interior_facet_domains);    
        for (uint i=0; i < data.A_macro_num_entries; i++)
          data.Ae_macro[i] += A_ufc.macro_A[i];
      }

      if (b_ufc.form.num_interior_facet_integrals() > 0)
      {
        compute_tensor_on_one_interior_facet(L, b_ufc, cell0, cell1, *facet, 
                                           b_coefficients, interior_facet_domains);  
        for (uint i=0; i < data.b_macro_num_entries; i++)
          data.be_macro[i] += b_ufc.macro_A[i];
      }

      // Get local favcet index
      const uint facet0 = cell0.index(*facet);
      const uint facet1 = cell1.index(*facet);

      // If we have local facet 0, compute cell contribution
      if (facet0 == 0)
      {
        if (A_ufc.form.num_cell_integrals() > 0) 
        {
          A_ufc.update(cell0);
          compute_tensor_on_one_cell(a, A_ufc, cell0, A_coefficients, cell_domains); 

          const uint nn = A_ufc.local_dimensions[0];
          const uint mm = A_ufc.local_dimensions[1];
          for (uint i = 0; i < mm; i++)
            for (uint j = 0; j < nn; j++)
              //A_ufc.macro_A[2*i*nn+j] += A_ufc.A[i*nn+j];
              data.Ae_macro[2*i*nn+j] += A_ufc.A[i*nn+j];
        }

        if (b_ufc.form.num_cell_integrals() > 0) 
        {
          b_ufc.update(cell0);
          compute_tensor_on_one_cell(L, b_ufc, cell0, b_coefficients, cell_domains); 
          for (uint i = 0; i < data.b_num_entries; i++)
            //b_ufc.macro_A[i] += b_ufc.A[i];
            data.be_macro[i] += b_ufc.A[i];
        }
      }

      // If we have local facet 0, compute cell contribution
      if (facet1 == 0)
      {
        if (A_ufc.form.num_cell_integrals() > 0) 
        {
          A_ufc.update(cell1);
          compute_tensor_on_one_cell(a, A_ufc, cell1, A_coefficients, cell_domains); 
          const uint nn = A_ufc.local_dimensions[0];
          const uint mm = A_ufc.local_dimensions[1];
          for (uint i=0; i < mm; i++)
            for (uint j=0; j < nn; j++)
              //A_ufc.macro_A[2*nn*mm + 2*i*nn + nn + j] += A_ufc.A[i*nn+j];
              data.Ae_macro[2*nn*mm + 2*i*nn + nn + j] += A_ufc.A[i*nn+j];
        }

        if (b_ufc.form.num_cell_integrals() > 0) 
        {
          b_ufc.update(cell1);
          compute_tensor_on_one_cell(L, b_ufc, cell1, b_coefficients, cell_domains); 
          for (uint i=0; i < data.b_num_entries; i++)
            //b_ufc.macro_A[i + data.b_num_entries] += b_ufc.A[i];
            data.be_macro[i + data.b_num_entries] += b_ufc.A[i];
        }
      }

      // Modify local matrix/element for Dirichlet boundary conditions
      apply_bc(data.Ae_macro, data.be_macro, data.indicators, data.g, 
               A_ufc.macro_dofs, A_ufc.macro_local_dimensions);
      //apply_bc(A_ufc.macro_A, b_ufc.macro_A, data.indicators, data.g, 
      //         A_ufc.macro_dofs, A_ufc.macro_local_dimensions);

      b_ufc.update(cell0, cell1);

      // FIXME: Figure out what to do with this
      // Tabulate dofs
      for (uint i = 0; i < A_ufc.form.rank(); i++)
      {
        const uint offset = A_ufc.local_dimensions[i];
        a.function_space(i).dofmap().tabulate_dofs(A_ufc.macro_dofs[i],
                                                   A_ufc.cell0, cell0.index());
        a.function_space(i).dofmap().tabulate_dofs(A_ufc.macro_dofs[i] + offset,
                                                   A_ufc.cell1, cell1.index());
      }
      const uint offset = b_ufc.local_dimensions[0];
      L.function_space(0).dofmap().tabulate_dofs(b_ufc.macro_dofs[0],
                                                 b_ufc.cell0, cell0.index());
      L.function_space(0).dofmap().tabulate_dofs(b_ufc.macro_dofs[0] + offset,
                                                 b_ufc.cell1, cell1.index());

      // Add entries to global tensor
      A.add(data.Ae_macro, A_ufc.macro_local_dimensions, A_ufc.macro_dofs);
      b.add(data.be_macro, b_ufc.macro_local_dimensions, b_ufc.macro_dofs);
      //A.add(A_ufc.macro_A, A_ufc.macro_local_dimensions, A_ufc.macro_dofs);
      //b.add(b_ufc.macro_A, b_ufc.macro_local_dimensions, b_ufc.macro_dofs);
    }

    // Exterior facet
    if ( facet->num_entities(mesh.topology().dim()) != 2 )
    {
      // Get mesh cell to which mesh facet belongs (pick first, there is only one)
      Cell cell(mesh, facet->entities(mesh.topology().dim())[0]);

      if (A_ufc.form.num_exterior_facet_integrals() > 0 )
        compute_tensor_on_one_exterior_facet(a, A_ufc, cell, *facet, 
                                             A_coefficients, 
                                             exterior_facet_domains);  

      if (b_ufc.form.num_exterior_facet_integrals() > 0 )
        compute_tensor_on_one_exterior_facet(L, b_ufc, cell, *facet, 
                                             b_coefficients, 
                                             exterior_facet_domains);  

      // If we have local facet 0, compute cell integral
      const uint local_facet = cell.index(*facet);
      if (local_facet == 0)
      {
        if (A_ufc.form.num_cell_integrals() > 0 )
          compute_tensor_on_one_cell(a, A_ufc, cell, A_coefficients, cell_domains); 

        if (b_ufc.form.num_cell_integrals() > 0 )
          compute_tensor_on_one_cell(L, b_ufc, cell, b_coefficients, cell_domains); 
      }

      // Modify local matrix/element for Dirichlet boundary conditions
      apply_bc(A_ufc.A, b_ufc.A, data.indicators, data.g, A_ufc.dofs, 
               A_ufc.local_dimensions);

      // Add entries to global tensor
      A.add(A_ufc.A, A_ufc.local_dimensions, A_ufc.dofs);
      b.add(b_ufc.A, b_ufc.local_dimensions, b_ufc.dofs);
    }
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::apply_bc(double* A, double* b, const uint* indicators, 
                               const double* g, uint** global_dofs, 
                               const uint* dims)
{
  const uint m = dims[0];
  const uint n = dims[1];

  for (uint i=0; i < n; i++)
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
  : A_num_entries(1), b_num_entries(1), Ae(0), be(0), Ae_macro(0), be_macro(0), 
    indicators(0), g(0) 
{
  for (uint i = 0; i < a.rank(); i++)
    A_num_entries *= a.function_space(i).dofmap().max_local_dimension();
  Ae = new double[A_num_entries];

  for (uint i = 0; i < L.rank(); i++)
    b_num_entries *= L.function_space(i).dofmap().max_local_dimension();
  be = new double[b_num_entries];
   
  //if (a.ufc_form().num_interior_facet_integrals() > 0)
  //{
    A_macro_num_entries = 4*A_num_entries;
    Ae_macro = new double[A_macro_num_entries];
  //}

  //if (L.ufc_form().num_interior_facet_integrals() > 0)
  //{
    b_macro_num_entries = 2*b_num_entries;
    be_macro = new double[b_macro_num_entries];
  //}

  const uint N = a.function_space(1).dofmap().global_dimension();  
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
  delete [] Ae_macro; 
  delete [] be_macro;
  delete [] indicators;
  delete [] g;
}
//-----------------------------------------------------------------------------
inline void SystemAssembler::Scratch::init_cell()
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
inline void SystemAssembler::Scratch::init_macro()
{
  if (Ae_macro)
  {
    for (uint i = 0; i < A_macro_num_entries; i++)
      Ae_macro[i] = 0.0;
  }
  if (be_macro)
  {
    for (uint i = 0; i < b_macro_num_entries; i++)
      be_macro[i] = 0.0;
  }
}
//-----------------------------------------------------------------------------
