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

  // Gather UFC  objects
  boost::array<UFC*, 2> ufc = { { &A_ufc, &b_ufc} } ;

  // Initialize global tensors
  if (A)
    init_global_tensor(*A, *_a);
  if (b)
    init_global_tensor(*b, *_L);

  // Gather tensors
  boost::array<GenericTensor*, 2> tensors = { {A, b} };

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
  if (!ufc[0]->form.has_interior_facet_integrals()
      && !ufc[1]->form.has_interior_facet_integrals())
  {
    // Assemble cell-wise (no interior facet integrals)
    cell_wise_assembly(tensors, ufc, data, boundary_values,
                       cell_domains, exterior_facet_domains, rescale);
  }
  else
  {
    // Facet-wise assembly is not working in parallel
    not_working_in_parallel("System assembly over interior facets");

    // Facet-wise assembly does not support subdomains
    for (std::size_t form = 0; form < 2; ++form)
    {
      if (ufc[form]->form.num_cell_domains() > 0 ||
          ufc[form]->form.num_exterior_facet_domains() > 0 ||
          ufc[form]->form.num_interior_facet_domains() > 0)
      {
        dolfin_error("SystemAssembler.cpp",
                     "assemble system",
                     "System assembler does not support forms containing "
                     "integrals over subdomains");
      }
    }

    // Assemble facet-wise (including cell assembly)
    facet_wise_assembly(tensors, ufc, data, boundary_values,
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
void
SystemAssembler::cell_wise_assembly(boost::array<GenericTensor*, 2>& tensors,
                                    boost::array<UFC*, 2>& ufc,
                                    Scratch& data,
                                    const DirichletBC::Map& boundary_values,
                                    const MeshFunction<std::size_t>* cell_domains,
                                    const MeshFunction<std::size_t>* exterior_facet_domains,
                                    const bool rescale)
{
  // Extract mesh
  const Mesh& mesh = ufc[0]->dolfin_form.mesh();

  // Initialize entities if using external facet integrals
  dolfin_assert(mesh.ordered());
  bool has_exterior_facet_integrals = ufc[0]->form.has_exterior_facet_integrals()
      || ufc[1]->form.has_exterior_facet_integrals();
  if (has_exterior_facet_integrals)
  {
    // Compute facets and facet-cell connectivity if not already computed
    const std::size_t D = mesh.topology().dim();
    mesh.init(D - 1);
    mesh.init(D - 1, D);
  }

  // Collect pointers to dof maps
  boost::array<std::vector<const GenericDofMap*>, 2> dofmaps;
  for (std::size_t i = 0; i < 2; ++i)
    dofmaps[0].push_back(ufc[0]->dolfin_form.function_space(i)->dofmap().get());
  dofmaps[1].push_back(ufc[1]->dolfin_form.function_space(0)->dofmap().get());

  // Vector to hold dof map for a cell
  boost::array<std::vector<const std::vector<dolfin::la_index>* >, 2> cell_dofs
    = { {std::vector<const std::vector<dolfin::la_index>* >(2),
         std::vector<const std::vector<dolfin::la_index>* >(1)} };

  // Create pointers to hold integral objects
  boost::array<const ufc::cell_integral*, 2> cell_integrals
    = { {ufc[0]->default_cell_integral.get(), ufc[1]->default_cell_integral.get()} };

  boost::array<const ufc::exterior_facet_integral*, 2> exterior_facet_integrals
    = { { ufc[0]->default_exterior_facet_integral.get(),
          ufc[1]->default_exterior_facet_integral.get()} };

  // Check whether integrals are domain-dependent
  bool use_cell_domains = cell_domains && !cell_domains->empty();
  bool use_exterior_facet_domains
    = exterior_facet_domains && !exterior_facet_domains->empty();

  // Iterate over all cells
  Progress p("Assembling system (cell-wise)", mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Loop over lhs and then rhs contributions
    for (std::size_t form = 0; form < 2; ++form)
    {
      // Get rank (lhs=2, rhs=1)
      const std::size_t rank = (form == 0) ? 2 : 1;

      // Zero data
      std::fill(data.Ae[form].begin(), data.Ae[form].end(), 0.0);

      // Get cell integrals for sub domain (if any)
      if (use_cell_domains)
      {
        const std::size_t domain = (*cell_domains)[*cell];
        cell_integrals[form] = ufc[form]->get_cell_integral(domain);
      }

      // Get local-to-global dof maps for cell
      for (std::size_t dim = 0; dim < rank; ++dim)
        cell_dofs[form][dim] = &(dofmaps[form][dim]->cell_dofs(cell->index()));

      // Compute cell tensor (if required)
      bool tensor_required = tensors[form];
      if (rank == 2)
      {
        dolfin_assert(cell_dofs[0][1]);
        tensor_required = cell_matrix_required(tensors[0], cell_integrals[0],
                                               boundary_values, *cell_dofs[0][1]);
      }

      if (tensor_required)
      {
        // Update to current cell
        ufc[form]->update(*cell);

        // Tabulate cell tensor
        cell_integrals[form]->tabulate_tensor(ufc[form]->A.data(), ufc[form]->w(),
                                              ufc[form]->cell.vertex_coordinates.data(),
                                              ufc[form]->cell.orientation);
        for (std::size_t i = 0; i < data.Ae[form].size(); ++i)
          data.Ae[form][i] += ufc[form]->A[i];
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
            exterior_facet_integrals[form]
              = ufc[form]->get_exterior_facet_integral(domain);
          }

          // Skip if there are no integrals
          if (!exterior_facet_integrals[form])
            continue;

          // Extract local facet index
          const std::size_t local_facet = cell->index(*facet);

          // Determine if tensor needs to be computed
          bool tensor_required = tensors[form];
          if (rank == 2)
          {
            dolfin_assert(cell_dofs[0][1]);
            tensor_required = cell_matrix_required(tensors[0],
                                                   exterior_facet_integrals[0],
                                                   boundary_values,
                                                   *cell_dofs[0][1]);
          }

          // Add exterior facet tensor
          if (tensor_required)
          {
            // Update to current cell
            ufc[form]->update(*cell, local_facet);

            // Tabulate exterior facet tensor
            exterior_facet_integrals[form]->tabulate_tensor(ufc[form]->A.data(),
                                                            ufc[form]->w(),
                                                            ufc[form]->cell.vertex_coordinates.data(),
                                                            local_facet);
            for (std::size_t i = 0; i < data.Ae[form].size(); i++)
            data.Ae[form][i] += ufc[form]->A[i];
          }
        }
      }
    }

    // Check dofmap is the same for LHS columns and RHS vector
    dolfin_assert(cell_dofs[1][0] == cell_dofs[0][1]);

    // Modify local matrix/element for Dirichlet boundary conditions
    apply_bc(data.Ae[0].data(), data.Ae[1].data(), boundary_values,
             *cell_dofs[0][0], *cell_dofs[0][1], rescale);

    // Add entries to global tensor
    for (std::size_t form = 0; form < 2; ++form)
    {
      if (tensors[form])
        tensors[form]->add(data.Ae[form].data(), cell_dofs[form]);
    }

    p++;
  }
}
//-----------------------------------------------------------------------------
void
SystemAssembler::facet_wise_assembly(boost::array<GenericTensor*, 2>& tensors,
                                     boost::array<UFC*, 2>& ufc,
                                     Scratch& data,
                                     const DirichletBC::Map& boundary_values,
                      const MeshFunction<std::size_t>* cell_domains,
                      const MeshFunction<std::size_t>* exterior_facet_domains,
                      const MeshFunction<std::size_t>* interior_facet_domains,
                      const bool rescale)
{
  // Extract mesh
  const Mesh& mesh = ufc[0]->dolfin_form.mesh();

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
  boost::array<std::vector<const GenericDofMap*>, 2> dofmaps;
  for (std::size_t i = 0; i < 2; ++i)
    dofmaps[0].push_back(ufc[0]->dolfin_form.function_space(i)->dofmap().get());
  dofmaps[1].push_back(ufc[1]->dolfin_form.function_space(0)->dofmap().get());

  // Cell dofmaps [form][cell][form dim]
  boost::array<boost::array<std::vector<const std::vector<dolfin::la_index>* >, 2 >, 2> cell_dofs;
  cell_dofs[0][0].resize(2);
  cell_dofs[0][1].resize(2);
  cell_dofs[1][0].resize(1);
  cell_dofs[1][1].resize(1);

  //  cell_dofs_a(2, std::vector<const std::vector<dolfin::la_index>* >(2));
  //std::vector<const std::vector<dolfin::la_index>* > cell_dofs_L(2);

  boost::array<Cell, 2> cell;
  boost::array<std::size_t, 2> cell_index;
  boost::array<std::size_t, 2> local_facet;

  // Vectors to hold dofs for macro cells
  boost::array<std::vector<std::vector<dolfin::la_index> >, 2> macro_dofs;
  macro_dofs[0].resize(2);
  macro_dofs[1].resize(1);

  // Holders for UFC integrals
  boost::array<const ufc::cell_integral*, 2> cell_integrals;
  boost::array<const ufc::exterior_facet_integral*, 2> exterior_facet_integrals;

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
          cell_dofs[0][c][d] = &(dofmaps[0][d]->cell_dofs(cell_index[c]));
          num_dofs_a[d] += cell_dofs[0][c][d]->size();
        }
        cell_dofs[1][c][0] = &(dofmaps[1][0]->cell_dofs(cell_index[c]));
        num_dofs_L += cell_dofs[1][c][0]->size();
      }

      // Resize macro dof vector
      macro_dofs[0][0].resize(num_dofs_a[0]);
      macro_dofs[0][1].resize(num_dofs_a[1]);
      macro_dofs[1][0].resize(num_dofs_L);

      // Cell integrals
      cell_integrals[0] = ufc[0]->default_cell_integral.get();
      cell_integrals[1] = ufc[1]->default_cell_integral.get();

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
          if (cell_integrals[0])
          {
            ufc[0]->update(cell[c]);
            cell_integrals[0]->tabulate_tensor(ufc[0]->A.data(), ufc[0]->w(),
                                               ufc[0]->cell.vertex_coordinates.data(),
                                               ufc[0]->cell.orientation);
            const std::size_t nn = cell_dofs[0][c][0]->size();
            const std::size_t mm = cell_dofs[0][c][1]->size();
            for (std::size_t i = 0; i < mm; i++)
            {
              for (std::size_t j = 0; j < nn; j++)
              {
                ufc[0]->macro_A[2*nn*mm*c + num_cells*i*nn + nn*c + j]
                  += ufc[0]->A[i*nn + j];
              }
            }
          }

          if (cell_integrals[1])
          {
            ufc[1]->update(cell[c]);
            cell_integrals[1]->tabulate_tensor(ufc[1]->A.data(), ufc[1]->w(),
                                               ufc[1]->cell.vertex_coordinates.data(),
                                               ufc[1]->cell.orientation);
            for (std::size_t i = 0; i < cell_dofs[1][c][0]->size(); i++)
              ufc[1]->macro_A[cell_dofs[1][c][0]->size()*c + i] += ufc[1]->A[i];
          }
        }
      }

      // Tabulate dofs on macro element
      for (std::size_t c = 0; c < num_cells; ++c)
      {
        // Tabulate dofs for each rank on macro element
        for (std::size_t d = 0; d < 2; ++d)
        {
          std::copy(cell_dofs[0][c][d]->begin(), cell_dofs[0][c][d]->end(),
                    macro_dofs[0][d].begin() + c*cell_dofs[0][c][d]->size());
        }
        std::copy(cell_dofs[1][c][0]->begin(), cell_dofs[1][c][0]->end(),
                  macro_dofs[1][0].begin() + c*cell_dofs[1][c][0]->size());
      }

      // Modify local tensor for bcs
      apply_bc(ufc[0]->macro_A.data(), ufc[1]->macro_A.data(), boundary_values,
               macro_dofs[0][0], macro_dofs[0][1], rescale);

      // Add entries to global tensor
      if (tensors[0])
        tensors[0]->add(ufc[0]->macro_A.data(), macro_dofs[0]);
      if (tensors[1])
        tensors[1]->add(ufc[1]->macro_A.data(), macro_dofs[1]);
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

      exterior_facet_integrals[0] = ufc[0]->default_exterior_facet_integral.get();
      if (exterior_facet_integrals[0])
      {
        ufc[0]->update(cell, local_facet);
        exterior_facet_integrals[0]->tabulate_tensor(ufc[0]->A.data(), ufc[0]->w(),
                                                    ufc[0]->cell.vertex_coordinates.data(),
                                                    local_facet);
        for (std::size_t i = 0; i < data.Ae[0].size(); i++)
          data.Ae[0][i] += ufc[0]->A[i];
      }

      exterior_facet_integrals[1] = ufc[1]->default_exterior_facet_integral.get();
      if (exterior_facet_integrals[1])
      {
        ufc[1]->update(cell, local_facet);
        exterior_facet_integrals[1]->tabulate_tensor(ufc[1]->A.data(), ufc[1]->w(),
                                                     ufc[1]->cell.vertex_coordinates.data(),
                                                     local_facet);
        for (std::size_t i = 0; i < data.Ae[1].size(); i++)
          data.Ae[1][i] += ufc[1]->A[i];
      }

      // If we have local facet 0, assemble cell integral
      if (local_facet == 0)
      {
        cell_integrals[0] = ufc[0]->default_cell_integral.get();
        if (cell_integrals[0])
        {
          ufc[0]->update(cell);
          cell_integrals[0]->tabulate_tensor(ufc[0]->A.data(), ufc[0]->w(),
                                             ufc[0]->cell.vertex_coordinates.data(),
                                             ufc[0]->cell.orientation);
          for (std::size_t i = 0; i < data.Ae[0].size(); i++)
            data.Ae[0][i] += ufc[0]->A[i];
        }

        cell_integrals[1] = ufc[1]->default_cell_integral.get();
        if (cell_integrals[1])
        {
          ufc[1]->update(cell);
          cell_integrals[1]->tabulate_tensor(ufc[1]->A.data(), ufc[1]->w(),
                                             ufc[1]->cell.vertex_coordinates.data(),
                                             ufc[1]->cell.orientation);
          for (std::size_t i = 0; i < data.Ae[1].size(); i++)
            data.Ae[1][i] += ufc[1]->A[i];
        }
      }

      // Tabulate dofs
      const std::size_t cell_index = cell.index();
      boost::array<std::vector<const std::vector<dolfin::la_index>* >, 2> dofs;
      dofs[0].resize(2);
      dofs[1].resize(1);
      dofs[0][0] =
        &(ufc[0]->dolfin_form.function_space(0)->dofmap()->cell_dofs(cell_index));
      dofs[0][1] =
        &(ufc[0]->dolfin_form.function_space(1)->dofmap()->cell_dofs(cell_index));
      dofs[1][0] =
        &(ufc[1]->dolfin_form.function_space(0)->dofmap()->cell_dofs(cell_index));

      // Modify local matrix/element for Dirichlet boundary conditions
      apply_bc(data.Ae[0].data(), data.Ae[1].data(), boundary_values,
               *dofs[0][0], *dofs[0][1], rescale);

      // Add entries to global tensor
      if (tensors[0])
        tensors[0]->add(data.Ae[0].data(), dofs[0]);
      if (tensors[1])
        tensors[1]->add(data.Ae[1].data(), dofs[1]);
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
         const std::vector<dolfin::la_index>& global_dofs0,
         const std::vector<dolfin::la_index>& global_dofs1,
         const bool rescale)
{
  dolfin_assert(A);
  dolfin_assert(b);

  // Wrap matrix and vector as Armadillo. Armadillo matrix storgae is
  // column-major, so all operations are transposed.
  arma::mat _A(A, global_dofs1.size(), global_dofs0.size(), false, true);
  arma::rowvec _b(b, global_dofs0.size(), false, true);

  // Loop over rows
  for (std::size_t i = 0; i < _A.n_rows; ++i)
  {
    const std::size_t ii = global_dofs1[i];
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
inline bool SystemAssembler::cell_matrix_required(const GenericTensor* A,
                                       const void* integral,
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
  Ae[0].resize(A_num_entries);
  Ae[1].resize(L.function_space(0)->dofmap()->max_cell_dimension());
}
//-----------------------------------------------------------------------------
SystemAssembler::Scratch::~Scratch()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
inline void SystemAssembler::Scratch::zero_cell()
{
  std::fill(Ae[0].begin(), Ae[0].end(), 0.0);
  std::fill(Ae[1].begin(), Ae[1].end(), 0.0);
}
//-----------------------------------------------------------------------------
