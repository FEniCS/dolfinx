// Copyright (C) 2008-2013 Kent-Andre Mardal and Garth N. Wells
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
// Modified by Anders Logg 2008-2013
// Modified by Joachim B Haga 2012
// Modified by Jan Blechta 2013
// Modified by Martin Alnaes 2013
//
// First added:  2009-06-22
// Last changed: 2013-04-18

#include <armadillo>
#include <boost/array.hpp>
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
SystemAssembler::SystemAssembler(const Form& a, const Form& L)
  : rescale(false), _a(reference_to_no_delete_pointer(a)),
    _L(reference_to_no_delete_pointer(L))
{
  // Check arity of forms
  check_arity(_a, _L);
}
//-----------------------------------------------------------------------------
SystemAssembler::SystemAssembler(const Form& a, const Form& L,
                                 const DirichletBC& bc)
  : rescale(false), _a(reference_to_no_delete_pointer(a)),
    _L(reference_to_no_delete_pointer(L))
{
  // Check arity of forms
  check_arity(_a, _L);

  // Store Dirichlet boundary condition
  _bcs.push_back(&bc);
}
//-----------------------------------------------------------------------------
SystemAssembler::SystemAssembler(const Form& a, const Form& L,
                                 const std::vector<const DirichletBC*> bcs)
  : rescale(false), _a(reference_to_no_delete_pointer(a)),
    _L(reference_to_no_delete_pointer(L)), _bcs(bcs)
{
  // Check arity of forms
  check_arity(_a, _L);
}
//-----------------------------------------------------------------------------
SystemAssembler::SystemAssembler(boost::shared_ptr<const Form> a,
                                 boost::shared_ptr<const Form> L)
  : rescale(false), _a(a), _L(L)
{
  // Check arity of forms
  check_arity(_a, _L);
}
//-----------------------------------------------------------------------------
SystemAssembler::SystemAssembler(boost::shared_ptr<const Form> a,
                                 boost::shared_ptr<const Form> L,
                                 const DirichletBC& bc)
  : rescale(false), _a(a), _L(L)
{
  // Check arity of forms
  check_arity(_a, _L);

  // Store Dirichlet boundary condition
  _bcs.push_back(&bc);
}
//-----------------------------------------------------------------------------
SystemAssembler::SystemAssembler(boost::shared_ptr<const Form> a,
                                 boost::shared_ptr<const Form> L,
                                 const std::vector<const DirichletBC*> bcs)
  : rescale(false), _a(a), _L(L), _bcs(bcs)
{
  // Check arity of forms
  check_arity(_a, _L);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(GenericMatrix& A, GenericVector& b)
{
  assemble(&A, &b, NULL);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(GenericMatrix& A)
{
  assemble(&A, NULL, NULL);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(GenericVector& b)
{
  assemble(NULL, &b, NULL);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(GenericMatrix& A, GenericVector& b,
                               const GenericVector& x0)
{
  assemble(&A, &b, &x0);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(GenericVector& b, const GenericVector& x0)
{
  assemble(NULL, &b, &x0);
}
//-----------------------------------------------------------------------------
void SystemAssembler::check_arity(boost::shared_ptr<const Form> a,
                                  boost::shared_ptr<const Form> L)
{
  // Check that a is a bilinear form
  if (a)
  {
    if (a->rank() != 2)
    {
      dolfin_error("SystemAssembler.cpp",
                   "assemble system",
                   "expected a bilinear form for a");
    }
  }

  // Check that a is a bilinear form
  if (L)
  {
    if (L->rank() != 1)
    {
      dolfin_error("SystemAssembler.cpp",
                   "assemble system",
                   "expected a linear form for L");
    }
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(GenericMatrix* A, GenericVector* b,
                               const GenericVector* x0)
{
  dolfin_assert(_a);
  dolfin_assert(_L);

  // Set timer
  Timer timer("Assemble system");

  // Get mesh
  const Mesh& mesh = _a->mesh();
  dolfin_assert(mesh.ordered());

  // Get cell domains
  const MeshFunction<std::size_t>* cell_domains = _a->cell_domains().get();
  if (cell_domains != _L->cell_domains().get())
  {
    warning("Bilinear and linear forms do not have same cell facet subdomains \
in SystemAssembler. Taking subdomains from bilinear form");
  }

  // Get exterior facet domains
  const MeshFunction<std::size_t>* exterior_facet_domains
    = _a->exterior_facet_domains().get();
  if (exterior_facet_domains != _L->exterior_facet_domains().get())
  {
    warning("Bilinear and linear forms do not have same exterior facet \
subdomains in SystemAssembler. Taking subdomains from bilinear form");
  }

  // Get interior facet domains
  const MeshFunction<std::size_t>* interior_facet_domains
    = _a->interior_facet_domains().get();
  if (interior_facet_domains != _L->interior_facet_domains().get())
  {
    warning("Bilinear and linear forms do not have same interior facet \
subdomains in SystemAssembler. Taking subdomains from bilinear form");
  }

  // Check forms
  AssemblerBase::check(*_a);
  AssemblerBase::check(*_L);

  // Check that we have a bilinear and a linear form
  dolfin_assert(_a->rank() == 2);
  dolfin_assert(_L->rank() == 1);

  // Check that forms share a function space
  if (*_a->function_space(1) != *_L->function_space(0))
  {
    dolfin_error("SystemAssembler.cpp",
                 "assemble system",
                 "expected forms (a, L) to share a FunctionSpace");
  }

  // FIXME: This may update coefficients twice. Checked for shared
  //        coefficients

  // Update off-process coefficients for a
  std::vector<boost::shared_ptr<const GenericFunction> > coefficients
    = _a->coefficients();
  for (std::size_t i = 0; i < coefficients.size(); ++i)
    coefficients[i]->update();

  // Update off-process coefficients for L
  coefficients = _L->coefficients();
  for (std::size_t i = 0; i < coefficients.size(); ++i)
    coefficients[i]->update();

  // Create data structures for local assembly data
  UFC A_ufc(*_a), b_ufc(*_L);
  boost::array<UFC*, 2> ufc;
  ufc[0] = &A_ufc;
  ufc[1] = &b_ufc;

  // Initialize global tensors
  if (A)
    init_global_tensor(*A, *_a);
  if (b)
    init_global_tensor(*b, *_L);

  // Allocate data
  Scratch data(*_a, *_L);

  // Get Dirichlet dofs and values for local mesh
  DirichletBC::Map boundary_values;
  for (std::size_t i = 0; i < _bcs.size(); ++i)
  {
    _bcs[i]->get_boundary_values(boundary_values);
    if (MPI::num_processes() > 1 && _bcs[i]->method() != "pointwise")
      _bcs[i]->gather(boundary_values);
  }

  // Modify boundary values for incremental (typically nonlinear)
  // problems
  if (x0)
  {
    if (MPI::num_processes() > 1)
    {
      warning("Parallel symmetric assembly over interior facets for nonlinear \
problems is untested");
    }
    dolfin_assert(x0->size()==_a->function_space(1)->dofmap()->global_dimension());

    const std::size_t num_bc_dofs = boundary_values.size();
    std::vector<dolfin::la_index> bc_indices;
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
    x0->get_local(x0_values.data(), num_bc_dofs, bc_indices.data());
    for (std::size_t i = 0; i < num_bc_dofs; i++)
      boundary_values[bc_indices[i]] = x0_values[i] - bc_values[i];
  }

  // Check whether we should do cell-wise or facet-wise assembly
  //if (!A_ufc.form.has_interior_facet_integrals()
  //    && !b_ufc.form.has_interior_facet_integrals())
  if (!ufc[0]->form.has_interior_facet_integrals()
      && !ufc[1]->form.has_interior_facet_integrals())
  {
    // Assemble cell-wise (no interior facet integrals)
    cell_wise_assembly(A, b, ufc, data, boundary_values,
                       cell_domains, exterior_facet_domains, rescale);
  }
  else
  {
    // Facet-wise assembly is not working in parallel
    not_working_in_parallel("System assembly over interior facets");

    // Facet-wise assembly does not support subdomains
    if (A_ufc.form.num_cell_domains() > 0 ||
        b_ufc.form.num_cell_domains() > 0 ||
        A_ufc.form.num_exterior_facet_domains() > 0 ||
        b_ufc.form.num_exterior_facet_domains() > 0 ||
        A_ufc.form.num_interior_facet_domains() > 0 ||
        b_ufc.form.num_interior_facet_domains() > 0)
    {
      dolfin_error("SystemAssembler.cpp",
                   "assemble system",
                   "System assembler does not support forms containing "
                   "integrals over subdomains");
    }

    // Assemble facet-wise (including cell assembly)
    facet_wise_assembly(A, b, ufc, data, boundary_values,
                        cell_domains, exterior_facet_domains,
                        interior_facet_domains, rescale);
  }

  // Finalise assembly
  if (finalize_tensor)
  {
    if (A)
      A->apply("add");
    if (b)
      b->apply("add");
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::cell_wise_assembly(GenericMatrix* A, GenericVector* b,
                       boost::array<UFC*, 2>& ufc,
                       Scratch& data,
                       const DirichletBC::Map& boundary_values,
                       const MeshFunction<std::size_t>* cell_domains,
                       const MeshFunction<std::size_t>* exterior_facet_domains,
                       const bool rescale)
{
  // FIXME: We can used some std::vectors or array pointers for the A
  // and b related terms to cut down on code repetition.

  // Extract mesh
  const Mesh& mesh = ufc[0]->dolfin_form.mesh();

  // Initialize entities if using external facet integrals
  dolfin_assert(mesh.ordered());
  bool has_exterior_facet_integrals = ufc[0]->form.has_exterior_facet_integrals()
      || ufc[1]->form.has_exterior_facet_integrals();
  if (has_exterior_facet_integrals)
  {
    // Compute facets and facet - cell connectivity if not already computed
    const std::size_t D = mesh.topology().dim();
    mesh.init(D - 1);
    mesh.init(D - 1, D);
  }

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> a_dofmaps;
  for (std::size_t i = 0; i < 2; ++i)
    a_dofmaps.push_back(ufc[0]->dolfin_form.function_space(i)->dofmap().get());

  std::vector<const GenericDofMap*> L_dofmaps;
  L_dofmaps.push_back(ufc[1]->dolfin_form.function_space(0)->dofmap().get());

  // Vector to hold dof map for a cell
  std::vector<const std::vector<dolfin::la_index>* > a_dofs(2);
  std::vector<const std::vector<dolfin::la_index>* > L_dofs(1);

  // Create pointers to hold integral objects
  const ufc::cell_integral* A_cell_integral = ufc[0]->default_cell_integral.get();
  const ufc::cell_integral* b_cell_integral = ufc[1]->default_cell_integral.get();
  const ufc::exterior_facet_integral* A_exterior_facet_integral
    = ufc[0]->default_exterior_facet_integral.get();
  const ufc::exterior_facet_integral* b_exterior_facet_integral
    = ufc[1]->default_exterior_facet_integral.get();

  // Check whether integrals are domain-dependent
  bool use_cell_domains = cell_domains && !cell_domains->empty();
  bool use_exterior_facet_domains
    = exterior_facet_domains && !exterior_facet_domains->empty();

  // Iterate over all cells
  Progress p("Assembling system (cell-wise)", mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // FIXME: Move this to avoid reset when not required
    // Reset cell tensor and vector
    std::fill(data.Ae.begin(), data.Ae.end(), 0.0);
    std::fill(data.be.begin(), data.be.end(), 0.0);

    // Get cell integrals for sub domain (if any)
    if (use_cell_domains)
    {
      const std::size_t domain = (*cell_domains)[*cell];
      A_cell_integral = ufc[0]->get_cell_integral(domain);
      b_cell_integral = ufc[1]->get_cell_integral(domain);
    }

    // Get local-to-global dof maps for cell
    a_dofs[0] = &(a_dofmaps[0]->cell_dofs(cell->index()));
    a_dofs[1] = &(a_dofmaps[1]->cell_dofs(cell->index()));
    L_dofs[0] = &(L_dofmaps[0]->cell_dofs(cell->index()));

    // Compute cell tensor for A (if required)
    dolfin_assert(a_dofs[1]);
    if (cell_matrix_required(A, A_cell_integral, boundary_values, *a_dofs[1]))
    {
      // Update to current cell
      ufc[0]->update(*cell);

      // Tabulate cell tensor
      A_cell_integral->tabulate_tensor(ufc[0]->A.data(), ufc[0]->w(),
                                       ufc[0]->cell.vertex_coordinates.data(),
                                       ufc[0]->cell.orientation);
      for (std::size_t i = 0; i < data.Ae.size(); ++i)
        data.Ae[i] += ufc[0]->A[i];
    }

    // Compute cell tensor for b
    if (b_cell_integral)
    {
      // Update to current cell
      ufc[1]->update(*cell);

      // Tabulate cell tensor
      b_cell_integral->tabulate_tensor(ufc[1]->A.data(), ufc[1]->w(),
                                       ufc[1]->cell.vertex_coordinates.data(),
                                       ufc[1]->cell.orientation);
      for (std::size_t i = 0; i < data.be.size(); ++i)
        data.be[i] += ufc[1]->A[i];
    }

    // Compute exterior facet integral if present
    if (has_exterior_facet_integrals)
    {
      for (FacetIterator facet(*cell); !facet.end(); ++facet)
      {
        // Only consider exterior facets
        if (!facet->exterior())
          continue;

        // Get exterior facet integrals for sub domain (if any)
        if (use_exterior_facet_domains)
        {
          const std::size_t domain = (*exterior_facet_domains)[*facet];
          A_exterior_facet_integral = ufc[0]->get_exterior_facet_integral(domain);
          b_exterior_facet_integral = ufc[1]->get_exterior_facet_integral(domain);
        }

        // Skip if there are no integrals
        if (!A_exterior_facet_integral && !b_exterior_facet_integral)
          continue;

        // Extract local facet index
        const std::size_t local_facet = cell->index(*facet);

        // Determine of Ae needs to be computed
        const bool compute_Ae = (A && A_exterior_facet_integral)
          || (A_exterior_facet_integral && has_bc(boundary_values,
                                        *(a_dofs[1])));

        // Add exterior facet tensor for A
        if (compute_Ae)
        {
          // Update to current cell
          ufc[0]->update(*cell, local_facet);

          // Tabulate exterior facet tensor
          A_exterior_facet_integral->tabulate_tensor(ufc[0]->A.data(),
                                                     ufc[0]->w(),
                           ufc[0]->cell.vertex_coordinates.data(), local_facet);
          for (std::size_t i = 0; i < data.Ae.size(); i++)
            data.Ae[i] += ufc[0]->A[i];
        }

        // Add exterior facet tensor for b
        if (b_exterior_facet_integral)
        {
          // Update to current cell
          ufc[1]->update(*cell, local_facet);

          // Tabulate exterior facet tensor
          b_exterior_facet_integral->tabulate_tensor(ufc[1]->A.data(),
                                                     ufc[1]->w(),
                                       ufc[1]->cell.vertex_coordinates.data(),
                                       local_facet);
          for (std::size_t i = 0; i < data.be.size(); i++)
            data.be[i] += ufc[1]->A[i];
        }
      }
    }

    dolfin_assert(L_dofs[0] == a_dofs[1]);

    // Modify local matrix/element for Dirichlet boundary conditions
    apply_bc(data.Ae.data(), data.be.data(), boundary_values, a_dofs, rescale);

    // Add entries to global tensor
    if (A)
      A->add(data.Ae.data(), a_dofs);
    if (b)
      b->add(data.be.data(), L_dofs);

    p++;
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::facet_wise_assembly(GenericMatrix* A, GenericVector* b,
                      boost::array<UFC*, 2>& ufc,
                      Scratch& data,
                      const DirichletBC::Map& boundary_values,
                      const MeshFunction<std::size_t>* cell_domains,
                      const MeshFunction<std::size_t>* exterior_facet_domains,
                      const MeshFunction<std::size_t>* interior_facet_domains,
                      const bool rescale)
{
  // Extract mesh and coefficients
  const Mesh& mesh = ufc[0]->dolfin_form.mesh();
  const std::vector<boost::shared_ptr<const GenericFunction> >
    A_coefficients = ufc[0]->dolfin_form.coefficients();
  const std::vector<boost::shared_ptr<const GenericFunction> >
    b_coefficients = ufc[1]->dolfin_form.coefficients();

  // Compute facets and facet - cell connectivity if not already computed
  const std::size_t D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);

  // Facet orientation not supported
  if (mesh.data().exists("facet_orientation", D - 1))
  {
    dolfin_error("SystemAssembler.cpp",
                 "assemble system",
                 "User-defined facet orientation is not supported by system \
assembler");
  }

  // Collect pointers to dof maps
  std::vector<const GenericDofMap*> a_dofmaps;
  for (std::size_t i = 0; i < 2; ++i)
    a_dofmaps.push_back(ufc[0]->dolfin_form.function_space(i)->dofmap().get());
  std::vector<const GenericDofMap*> L_dofmaps;
  L_dofmaps.push_back(ufc[1]->dolfin_form.function_space(0)->dofmap().get());

  // Cell dofmaps
  std::vector<std::vector<const std::vector<dolfin::la_index>* > >
    cell_dofs_a(2, std::vector<const std::vector<dolfin::la_index>* >(2));
  std::vector<const std::vector<dolfin::la_index>* > cell_dofs_L(2);

  std::vector<Cell> cell(2);
  std::vector<std::size_t> cell_index(2);
  std::vector<std::size_t> local_facet(2);

  // Vectors to hold dofs for macro cells
  std::vector<std::vector<dolfin::la_index> > a_macro_dofs(2);
  std::vector<std::vector<dolfin::la_index> > L_macro_dofs(1);

  // Iterate over facets
  Progress p("Assembling system (facet-wise)", mesh.num_facets());
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Number of cells sharing facet
    const std::size_t num_cells = facet->num_entities(mesh.topology().dim());

    // Interior facet
    if (num_cells == 2)
    {
      for (std::size_t c = 0; c < num_cells; ++c)
      {
        // Get cells incident with facet and update UFC objects
        cell[c] = Cell(mesh, facet->entities(mesh.topology().dim())[c]);

        // Cell indices
        cell_index[c] = cell[c].index();

        // Get local facet indices
        local_facet[c] = cell[c].index(*facet);
      }

      // Update UFC objects
      ufc[0]->update(cell[0], local_facet[0], cell[1], local_facet[1]);
      ufc[1]->update(cell[0], local_facet[0], cell[1], local_facet[1]);

      // Reset some temp data
      std::fill(ufc[0]->macro_A.begin(), ufc[0]->macro_A.end(), 0.0);
      std::fill(ufc[1]->macro_A.begin(), ufc[1]->macro_A.end(), 0.0);

      // Tabulate dofs for cell0 and cell1
      std::size_t num_dofs_a[2] = {0, 0};
      std::size_t num_dofs_L = 0;
      for (std::size_t c = 0; c < num_cells; ++c)
      {
        for (int d = 0; d < 2; ++d)
        {
          cell_dofs_a[c][d]
            = &(ufc[0]->dolfin_form.function_space(d)->dofmap()->cell_dofs(cell_index[c]));
          num_dofs_a[d] +=  cell_dofs_a[c][d]->size();
        }
        cell_dofs_L[c]
          = &(ufc[1]->dolfin_form.function_space(0)->dofmap()->cell_dofs(cell_index[c]));
        num_dofs_L += cell_dofs_L[c]->size();
      }

      // Resize macro dof vector
      a_macro_dofs[0].resize(num_dofs_a[0]);
      a_macro_dofs[1].resize(num_dofs_a[1]);
      L_macro_dofs[0].resize(num_dofs_L);

      // Cell integrals
      const ufc::cell_integral* A_cell_integral
        = ufc[0]->default_cell_integral.get();
      const ufc::cell_integral* b_cell_integral
        = ufc[1]->default_cell_integral.get();

      // Compute facet contribution to A
      if (num_cells == 2 && ufc[0]->form.has_interior_facet_integrals())
      {
        compute_tensor_on_one_interior_facet(ufc[0]->dolfin_form,
                                             *ufc[0], cell[0], cell[1],
                                             *facet, NULL);
      }

      // Compute facet contribution to b
      if (num_cells == 2 && ufc[1]->form.has_interior_facet_integrals())
      {
        compute_tensor_on_one_interior_facet(ufc[1]->dolfin_form,
                                             *ufc[1], cell[0], cell[1],
                                             *facet, NULL);
      }

      // If we have local facet 0 for cell[i], compute cell contribution
      for (std::size_t c = 0; c < num_cells; ++c)
      {
        if (local_facet[c] == 0)
        {
          if (A_cell_integral)
          {
            ufc[0]->update(cell[c]);
            A_cell_integral->tabulate_tensor(ufc[0]->A.data(), ufc[0]->w(),
                                          ufc[0]->cell.vertex_coordinates.data(),
                                          ufc[0]->cell.orientation);
            const std::size_t nn = cell_dofs_a[c][0]->size();
            const std::size_t mm = cell_dofs_a[c][1]->size();
            for (std::size_t i = 0; i < mm; i++)
            {
              for (std::size_t j = 0; j < nn; j++)
              {
                ufc[0]->macro_A[2*nn*mm*c + num_cells*i*nn + nn*c + j]
                  += ufc[0]->A[i*nn + j];
              }
            }
          }

          if (b_cell_integral)
          {
            ufc[1]->update(cell[c]);
            b_cell_integral->tabulate_tensor(ufc[1]->A.data(), ufc[1]->w(),
                                           ufc[1]->cell.vertex_coordinates.data(),
                                           ufc[1]->cell.orientation);
            for (std::size_t i = 0; i < cell_dofs_L[c]->size(); i++)
              ufc[1]->macro_A[cell_dofs_L[c]->size()*c + i] += ufc[1]->A[i];
          }
        }
      }

      // Tabulate dofs on macro element
      for (std::size_t c = 0; c < num_cells; ++c)
      {
        // Tabulate dofs for each rank on macro element
        for (std::size_t d = 0; d < 2; ++d)
        {
          std::copy(cell_dofs_a[c][d]->begin(), cell_dofs_a[c][d]->end(),
                    a_macro_dofs[d].begin() + c*cell_dofs_a[c][d]->size());
        }
        std::copy(cell_dofs_L[c]->begin(), cell_dofs_L[c]->end(),
                  L_macro_dofs[0].begin() + c*cell_dofs_L[c]->size());
      }

      // Modify local matrix/element for Dirichlet boundary conditions
      std::vector<const std::vector<dolfin::la_index>* > _a_macro_dofs(2);
      _a_macro_dofs[0] = &a_macro_dofs[0];
      _a_macro_dofs[1] = &a_macro_dofs[1];

      apply_bc(ufc[0]->macro_A.data(), ufc[1]->macro_A.data(), boundary_values,
               _a_macro_dofs, rescale);

      // Add entries to global tensor
      if (A)
        A->add(ufc[0]->macro_A.data(), a_macro_dofs);
      if (b)
        b->add(ufc[1]->macro_A.data(), L_macro_dofs);
    }
    else // Exterior facet
    {
      // Get mesh cell to which mesh facet belongs (pick first, there
      // is only one)
      Cell cell(mesh, facet->entities(mesh.topology().dim())[0]);

      // Get local index of facet with respect to the cell
      const std::size_t local_facet = cell.index(*facet);

      // Update UFC objects
      ufc[0]->update(cell, local_facet);
      ufc[1]->update(cell, local_facet);

      // Reset some temp data
      std::fill(ufc[0]->A.begin(), ufc[0]->A.end(), 0.0);
      std::fill(ufc[1]->A.begin(), ufc[1]->A.end(), 0.0);

      // Initialize macro element matrix/vector to zero
      data.zero_cell();

      // Assemble exterior facet and attached cells if needed
      //assemble_exterior_facet(A, b, A_ufc, b_ufc, a, L, cell, *facet, data,
      //                       boundary_values, rescale);

      ufc::exterior_facet_integral* A_facet_integral
        = ufc[0]->default_exterior_facet_integral.get();
      if (A_facet_integral)
      {
        ufc[0]->update(cell, local_facet);
        A_facet_integral->tabulate_tensor(ufc[0]->A.data(), ufc[0]->w(),
                                          ufc[0]->cell.vertex_coordinates.data(),
                                          local_facet);
        for (std::size_t i = 0; i < data.Ae.size(); i++)
          data.Ae[i] += ufc[0]->A[i];
      }

      const ufc::exterior_facet_integral* b_facet_integral
        = ufc[1]->default_exterior_facet_integral.get();
      if (b_facet_integral)
      {
        ufc[1]->update(cell, local_facet);
        b_facet_integral->tabulate_tensor(ufc[1]->A.data(), ufc[1]->w(),
                                          ufc[1]->cell.vertex_coordinates.data(),
                                          local_facet);
        for (std::size_t i = 0; i < data.be.size(); i++)
          data.be[i] += ufc[1]->A[i];
      }

      // If we have local facet 0, assemble cell integral
      if (local_facet == 0)
      {
        const ufc::cell_integral* A_cell_integral
          = ufc[0]->default_cell_integral.get();
        if (A_cell_integral)
        {
          ufc[0]->update(cell);
          A_cell_integral->tabulate_tensor(ufc[0]->A.data(), ufc[0]->w(),
                                           ufc[0]->cell.vertex_coordinates.data(),
                                           ufc[0]->cell.orientation);
          for (std::size_t i = 0; i < data.Ae.size(); i++)
            data.Ae[i] += ufc[0]->A[i];
        }

        const ufc::cell_integral* b_cell_integral
          = ufc[1]->default_cell_integral.get();
        if (b_cell_integral)
        {
          ufc[1]->update(cell);
          b_cell_integral->tabulate_tensor(ufc[1]->A.data(), ufc[1]->w(),
                                           ufc[1]->cell.vertex_coordinates.data(),
                                           ufc[1]->cell.orientation);
          for (std::size_t i = 0; i < data.be.size(); i++)
            data.be[i] += ufc[1]->A[i];
        }
      }

      // Tabulate dofs
      const std::size_t cell_index = cell.index();
      std::vector<const std::vector<dolfin::la_index>* > a_dofs(2);
      std::vector<const std::vector<dolfin::la_index>* > L_dofs(1);
      a_dofs[0] =
        &(ufc[0]->dolfin_form.function_space(0)->dofmap()->cell_dofs(cell_index));
      a_dofs[1] =
        &(ufc[0]->dolfin_form.function_space(1)->dofmap()->cell_dofs(cell_index));
      L_dofs[0] =
        &(ufc[1]->dolfin_form.function_space(0)->dofmap()->cell_dofs(cell_index));

      // Modify local matrix/element for Dirichlet boundary conditions
      apply_bc(data.Ae.data(), data.be.data(), boundary_values, a_dofs, rescale);

      // Add entries to global tensor
      if (A)
        A->add(data.Ae.data(), a_dofs);
      if (b)
        b->add(data.be.data(), L_dofs);
    }
    p++;
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::compute_tensor_on_one_interior_facet(const Form& a,
            UFC& ufc, const Cell& cell0, const Cell& cell1, const Facet& facet,
            const MeshFunction<std::size_t>* interior_facet_domains)
{
  const std::vector<boost::shared_ptr<const GenericFunction> >
    coefficients = a.coefficients();

  // Facet integral
  ufc::interior_facet_integral* interior_facet_integral
    = ufc.default_interior_facet_integral.get();

  // Get integral for sub domain (if any)
  if (interior_facet_domains && !interior_facet_domains->empty())
  {
    const std::size_t domain = (*interior_facet_domains)[facet];
    interior_facet_integral = ufc.get_interior_facet_integral(domain);
  }

  // Get local index of facet with respect to each cell
  const std::size_t local_facet0 = cell0.index(facet);
  const std::size_t local_facet1 = cell1.index(facet);

  // Update to current pair of cells
  ufc.update(cell0, local_facet0, cell1, local_facet1);

  // Integrate over facet
  interior_facet_integral->tabulate_tensor(ufc.macro_A.data(), ufc.macro_w(),
                                           ufc.cell0.vertex_coordinates.data(),
                                           ufc.cell1.vertex_coordinates.data(),
                                           local_facet0, local_facet1);
}
//-----------------------------------------------------------------------------
inline void SystemAssembler::apply_bc(double* A, double* b,
         const DirichletBC::Map& boundary_values,
         const std::vector<const std::vector<dolfin::la_index>* >& global_dofs,
         const bool rescale)
{
  dolfin_assert(A);
  dolfin_assert(b);

  // Wrap matrix and vector as Armadillo. Armadillo matrix storgae is
  // column-major, so all operations are transposed.
  arma::mat _A(A, global_dofs[1]->size(), global_dofs[0]->size(), false, true);
  arma::rowvec _b(b, global_dofs[0]->size(), false, true);

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
      if (!rescale || std::abs(bc_value->second) < (b_norm + DOLFIN_EPS))
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
bool SystemAssembler::has_bc(const DirichletBC::Map& boundary_values,
                             const std::vector<dolfin::la_index>& dofs)
{
  // Loop over dofs and check if bc is applied
  std::vector<dolfin::la_index>::const_iterator dof;
  for (dof = dofs.begin(); dof != dofs.end(); ++dof)
  {
    DirichletBC::Map::const_iterator bc_value = boundary_values.find(*dof);
    if (bc_value != boundary_values.end())
      return true;
  }

  return false;
}
//-----------------------------------------------------------------------------
inline bool SystemAssembler::cell_matrix_required(const GenericMatrix* A,
                                       const ufc::cell_integral* integral,
                                       const DirichletBC::Map& boundary_values,
                                       const std::vector<dolfin::la_index>& dofs)
{
  if (A && integral)
    return true;
  else if (integral && has_bc(boundary_values, dofs))
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
SystemAssembler::Scratch::Scratch(const Form& a, const Form& L)
{
  std::size_t A_num_entries
    = a.function_space(0)->dofmap()->max_cell_dimension();
  A_num_entries *= a.function_space(1)->dofmap()->max_cell_dimension();
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
