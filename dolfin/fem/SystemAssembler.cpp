// Copyright (C) 2008-2015 Kent-Andre Mardal and Garth N. Wells
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
// Modified by Martin Alnaes 2013-2015

#include <array>
#include <algorithm>
#include <Eigen/Dense>
#include <dolfin/common/ArrayView.h>
#include <dolfin/common/Timer.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericMatrix.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
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
  : _a(reference_to_no_delete_pointer(a)),
    _l(reference_to_no_delete_pointer(L))
{
  // Check arity of forms
  check_arity(_a, _l);
}
//-----------------------------------------------------------------------------
SystemAssembler::SystemAssembler(const Form& a, const Form& L,
                                 const DirichletBC& bc)
  : _a(reference_to_no_delete_pointer(a)),
    _l(reference_to_no_delete_pointer(L))
{
  // Check arity of forms
  check_arity(_a, _l);

  // Store Dirichlet boundary condition
  _bcs.push_back(&bc);
}
//-----------------------------------------------------------------------------
SystemAssembler::SystemAssembler(const Form& a, const Form& L,
                                 const std::vector<const DirichletBC*> bcs)
  : _a(reference_to_no_delete_pointer(a)),
    _l(reference_to_no_delete_pointer(L)), _bcs(bcs)
{
  // Check arity of forms
  check_arity(_a, _l);
}
//-----------------------------------------------------------------------------
SystemAssembler::SystemAssembler(std::shared_ptr<const Form> a,
                                 std::shared_ptr<const Form> L)
  : _a(a), _l(L)
{
  // Check arity of forms
  check_arity(_a, _l);
}
//-----------------------------------------------------------------------------
SystemAssembler::SystemAssembler(std::shared_ptr<const Form> a,
                                 std::shared_ptr<const Form> L,
                                 const DirichletBC& bc)
  : _a(a), _l(L)
{
  // Check arity of forms
  check_arity(_a, _l);

  // Store Dirichlet boundary condition
  _bcs.push_back(&bc);
}
//-----------------------------------------------------------------------------
SystemAssembler::SystemAssembler(std::shared_ptr<const Form> a,
                                 std::shared_ptr<const Form> L,
                                 const std::vector<const DirichletBC*> bcs)
  : _a(a), _l(L), _bcs(bcs)
{
  // Check arity of forms
  check_arity(_a, _l);
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
void SystemAssembler::check_arity(std::shared_ptr<const Form> a,
                                  std::shared_ptr<const Form> L)
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
std::shared_ptr<const MeshFunction<std::size_t> >
_pick_one_meshfunction(std::string name,
                       std::shared_ptr<const MeshFunction<std::size_t> > a,
                       std::shared_ptr<const MeshFunction<std::size_t> > b)
{
  if ((a && b) && a != b)
  {
      warning("Bilinear and linear forms do not have same %s subdomains \
in SystemAssembler. Taking %s subdomains from bilinear form",
              name.c_str(), name.c_str());
  }
  return a ? a: b;
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(GenericMatrix* A, GenericVector* b,
                               const GenericVector* x0)
{
  dolfin_assert(_a);
  dolfin_assert(_l);

  // Set timer
  Timer timer("Assemble system");

  // Get mesh
  const Mesh& mesh = _a->mesh();
  dolfin_assert(mesh.ordered());

  // Get cell domains
  std::shared_ptr<const MeshFunction<std::size_t> > cell_domains
    = _pick_one_meshfunction("cell_domains", _a->cell_domains(),
                             _l->cell_domains());

  // Get exterior facet domains
  std::shared_ptr<const MeshFunction<std::size_t> > exterior_facet_domains
    = _pick_one_meshfunction("exterior_facet_domains",
                             _a->exterior_facet_domains(),
                             _l->exterior_facet_domains());

  // Get interior facet domains
  std::shared_ptr<const MeshFunction<std::size_t> > interior_facet_domains
    = _pick_one_meshfunction("interior_facet_domains",
                             _a->interior_facet_domains(),
                             _l->interior_facet_domains());

  // Check forms
  AssemblerBase::check(*_a);
  AssemblerBase::check(*_l);

  // Check that we have a bilinear and a linear form
  dolfin_assert(_a->rank() == 2);
  dolfin_assert(_l->rank() == 1);

  // Check that forms share a function space
  if (*_a->function_space(1) != *_l->function_space(0))
  {
    dolfin_error("SystemAssembler.cpp",
                 "assemble system",
                 "expected forms (a, L) to share a FunctionSpace");
  }

  // Create data structures for local assembly data
  UFC A_ufc(*_a), b_ufc(*_l);

  // Raise error for Point integrals
  if (A_ufc.form.has_point_integrals() || b_ufc.form.has_point_integrals())
  {
    dolfin_error("SystemAssembler.cpp",
                 "assemble system",
                 "Point integrals are not supported (yet)");
  }

  // Gather UFC  objects
  std::array<UFC*, 2> ufc = { { &A_ufc, &b_ufc} } ;

  // Initialize global tensors
  if (A)
    init_global_tensor(*A, *_a);
  if (b)
    init_global_tensor(*b, *_l);

  // Gather tensors
  std::array<GenericTensor*, 2> tensors = { {A, b} };

  // Allocate data
  Scratch data(*_a, *_l);

  // Get Dirichlet dofs and values for local mesh
  DirichletBC::Map boundary_values;
  for (std::size_t i = 0; i < _bcs.size(); ++i)
  {
    _bcs[i]->get_boundary_values(boundary_values);
    if (MPI::size(mesh.mpi_comm()) > 1 && _bcs[i]->method() != "pointwise")
      _bcs[i]->gather(boundary_values);
  }

  // Modify boundary values for incremental (typically nonlinear)
  // problems
  if (x0)
  {
    dolfin_assert(x0->size()
                  == _a->function_space(1)->dofmap()->global_dimension());

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
                       cell_domains, exterior_facet_domains);
  }
  else
  {
    // Assemble facet-wise (including cell assembly)
    facet_wise_assembly(tensors, ufc, data, boundary_values,
                        cell_domains, exterior_facet_domains,
                        interior_facet_domains);
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
SystemAssembler::cell_wise_assembly(std::array<GenericTensor*, 2>& tensors,
                std::array<UFC*, 2>& ufc,
                Scratch& data,
                const DirichletBC::Map& boundary_values,
                std::shared_ptr<const MeshFunction<std::size_t> > cell_domains,
                std::shared_ptr<const MeshFunction<std::size_t> > exterior_facet_domains)
{
  // Extract mesh
  const Mesh& mesh = ufc[0]->dolfin_form.mesh();

  // Initialize entities if using external facet integrals
  dolfin_assert(mesh.ordered());
  bool has_exterior_facet_integrals=ufc[0]->form.has_exterior_facet_integrals()
      || ufc[1]->form.has_exterior_facet_integrals();
  if (has_exterior_facet_integrals)
  {
    // Compute facets and facet-cell connectivity if not already computed
    const std::size_t D = mesh.topology().dim();
    mesh.init(D - 1);
    mesh.init(D - 1, D);
  }

  // Collect pointers to dof maps
  std::array<std::vector<const GenericDofMap*>, 2> dofmaps;
  for (std::size_t i = 0; i < 2; ++i)
    dofmaps[0].push_back(ufc[0]->dolfin_form.function_space(i)->dofmap().get());
  dofmaps[1].push_back(ufc[1]->dolfin_form.function_space(0)->dofmap().get());

  // Vector to hold dof map for a cell
  std::array<std::vector<ArrayView<const dolfin::la_index>>, 2> cell_dofs
    = { {std::vector<ArrayView<const dolfin::la_index>>(2),
         std::vector<ArrayView<const dolfin::la_index>>(1)} };

  // Create pointers to hold integral objects
  std::array<const ufc::cell_integral*, 2> cell_integrals
    = { {ufc[0]->default_cell_integral.get(),
         ufc[1]->default_cell_integral.get()} };

  std::array<const ufc::exterior_facet_integral*, 2> exterior_facet_integrals
    = { { ufc[0]->default_exterior_facet_integral.get(),
          ufc[1]->default_exterior_facet_integral.get()} };

  // Check whether integrals are domain-dependent
  bool use_cell_domains = cell_domains && !cell_domains->empty();
  bool use_exterior_facet_domains
    = exterior_facet_domains && !exterior_facet_domains->empty();

  // Iterate over all cells
  ufc::cell ufc_cell;
  std::vector<double> vertex_coordinates;
  Progress p("Assembling system (cell-wise)", mesh.num_cells());
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    // Check that cell is not a ghost
    dolfin_assert(!cell->is_ghost());

    // Get cell vertex coordinates
    cell->get_vertex_coordinates(vertex_coordinates);

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
        cell_dofs[form][dim] = dofmaps[form][dim]->cell_dofs(cell->index());

      // Compute cell tensor (if required)
      bool tensor_required;
      if (rank == 2) // form == 0
      {
        tensor_required = cell_matrix_required(tensors[form],
                                               cell_integrals[form],
                                               boundary_values,
                                               cell_dofs[form][1]);
      }
      else
        tensor_required = tensors[form] && cell_integrals[form];

      if (tensor_required)
      {
        // Update to current cell
        cell->get_cell_data(ufc_cell);
        ufc[form]->update(*cell, vertex_coordinates, ufc_cell,
                          cell_integrals[form]->enabled_coefficients());

        // Tabulate cell tensor
        cell_integrals[form]->tabulate_tensor(ufc[form]->A.data(),
                                              ufc[form]->w(),
                                              vertex_coordinates.data(),
                                              ufc_cell.orientation);
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
          bool tensor_required;
          if (rank == 2) // form == 0
          {
            tensor_required
              = cell_matrix_required(tensors[form],
                                     exterior_facet_integrals[form],
                                     boundary_values,
                                     cell_dofs[form][1]);
          }
          else
            tensor_required = tensors[form];

          // Add exterior facet tensor
          if (tensor_required)
          {
            // Update to current cell
            cell->get_cell_data(ufc_cell);
            ufc[form]->update(*cell, vertex_coordinates, ufc_cell,
                              exterior_facet_integrals[form]->enabled_coefficients());

            // Tabulate exterior facet tensor
            exterior_facet_integrals[form]->tabulate_tensor(ufc[form]->A.data(),
                                                            ufc[form]->w(),
                                                            vertex_coordinates.data(),
                                                            local_facet,
                                                            ufc_cell.orientation);
            for (std::size_t i = 0; i < data.Ae[form].size(); i++)
              data.Ae[form][i] += ufc[form]->A[i];
          }
        }
      }
    }

    // Check dofmap is the same for LHS columns and RHS vector
    dolfin_assert(cell_dofs[1][0].data() == cell_dofs[0][1].data());

    // Modify local matrix/element for Dirichlet boundary conditions
    apply_bc(data.Ae[0].data(), data.Ae[1].data(), boundary_values,
             cell_dofs[0][0], cell_dofs[0][1]);

    // Add entries to global tensor
    for (std::size_t form = 0; form < 2; ++form)
    {
      if (tensors[form])
        tensors[form]->add_local(data.Ae[form].data(), cell_dofs[form]);
    }

    p++;
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::facet_wise_assembly(
  std::array<GenericTensor*, 2>& tensors,
  std::array<UFC*, 2>& ufc,
  Scratch& data,
  const DirichletBC::Map& boundary_values,
  std::shared_ptr<const MeshFunction<std::size_t> > cell_domains,
  std::shared_ptr<const MeshFunction<std::size_t> > exterior_facet_domains,
  std::shared_ptr<const MeshFunction<std::size_t> > interior_facet_domains)
{
  // Extract mesh
  const Mesh& mesh = ufc[0]->dolfin_form.mesh();

  // Compute facets and facet - cell connectivity if not already computed
  const std::size_t D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);

  // My MPI rank
  const int my_mpi_rank = MPI::rank(mesh.mpi_comm());

  // Collect pointers to dof maps
  std::array<std::vector<const GenericDofMap*>, 2> dofmaps;
  for (std::size_t i = 0; i < 2; ++i)
    dofmaps[0].push_back(ufc[0]->dolfin_form.function_space(i)->dofmap().get());
  dofmaps[1].push_back(ufc[1]->dolfin_form.function_space(0)->dofmap().get());

  // Cell dofmaps [form][cell][form dim]
  std::array<std::array<std::vector<ArrayView<const dolfin::la_index>>,
                        2 >, 2> cell_dofs;
  cell_dofs[0][0].resize(2);
  cell_dofs[0][1].resize(2);
  cell_dofs[1][0].resize(1);
  cell_dofs[1][1].resize(1);

  std::array<Cell, 2> cell;
  std::array<std::size_t, 2> cell_index;
  std::array<std::size_t, 2> local_facet;

  // Vectors to hold dofs for macro cells
  std::array<std::vector<std::vector<dolfin::la_index>>, 2> macro_dofs;
  macro_dofs[0].resize(2);
  macro_dofs[1].resize(1);

  // Holder for number of dofs in macro-dofmap
  std::vector<std::size_t> num_dofs(2);

  // Holders for UFC integrals
  std::array<const ufc::cell_integral*, 2> cell_integrals
    = { { ufc[0]->default_cell_integral.get(),
          ufc[1]->default_cell_integral.get() } };
  std::array<const ufc::exterior_facet_integral*, 2> exterior_facet_integrals
    = { { ufc[0]->default_exterior_facet_integral.get(),
          ufc[1]->default_exterior_facet_integral.get() } };
  std::array<const ufc::interior_facet_integral*, 2> interior_facet_integrals
    = { { ufc[0]->default_interior_facet_integral.get(),
          ufc[1]->default_interior_facet_integral.get() } };

  // Check whether integrals are domain-dependent
  bool use_cell_domains = cell_domains && !cell_domains->empty();
  bool use_interior_facet_domains
    = interior_facet_domains && !interior_facet_domains->empty();
  bool use_exterior_facet_domains
    = exterior_facet_domains && !exterior_facet_domains->empty();

  // Indicator whether or not tensor is required
  std::array<bool, 2> tensor_required_cell, tensor_required_facet;

  // Iterate over facets
  std::array<ufc::cell, 2> ufc_cell;
  std::array<std::vector<double>, 2> vertex_coordinates;
  Progress p("Assembling system (facet-wise)", mesh.num_facets());
  for (FacetIterator facet(mesh); !facet.end(); ++facet)
  {
    // Number of cells sharing facet
    const std::size_t num_cells = facet->num_entities(D);

    // Check that facet is not a ghost
    dolfin_assert(!facet->is_ghost());

    // Interior facet
    if (num_cells == 2)
    {
      // Get cells incident with facet (which is 0 and 1 here is arbitrary)
      dolfin_assert(facet->num_entities(D) == 2);
      std::array<std::size_t, 2> cell_indices = { {
          facet->entities(D)[0],
          facet->entities(D)[1]
            } };

      // Make sure cell marker for + side is larger than cell marker for - side.
      // Note: by ffc convention, 0 is + and 1 is -
      if (use_cell_domains && (*cell_domains)[cell_indices[0]] < (*cell_domains)[cell_indices[1]])
        std::swap(cell_indices[0], cell_indices[1]);

      // Get cells incident with facet and associated data
      for (std::size_t c = 0; c < 2; ++c)
      {
        cell[c] = Cell(mesh, cell_indices[c]);
        cell_index[c] = cell[c].index();
        local_facet[c] = cell[c].index(*facet);
        cell[c].get_vertex_coordinates(vertex_coordinates[c]);
        cell[c].get_cell_data(ufc_cell[c], local_facet[c]);
      }

      const bool process_facet = (cell[0].is_ghost() != cell[1].is_ghost());
      bool facet_owner = true;
      if (process_facet)
      {
        int ghost_rank = -1;
        if (cell[0].is_ghost())
          ghost_rank = cell[0].owner();
        else
          ghost_rank = cell[1].owner();
        dolfin_assert(my_mpi_rank != ghost_rank);
        dolfin_assert(ghost_rank != -1);
        if (ghost_rank < my_mpi_rank)
          facet_owner = false;
      }

      // Loop over lhs and then rhs contributions
      for (std::size_t form = 0; form < 2; ++form)
      {
        // Get rank (lhs=2, rhs=1)
        const std::size_t rank = (form == 0) ? 2 : 1;

        // Compute number of dofs in macro dofmap
        std::fill(num_dofs.begin(), num_dofs.begin() + rank, 0);
        for (std::size_t c = 0; c < num_cells; ++c)
        {
          for (std::size_t dim = 0; dim < rank; ++dim)
          {
            cell_dofs[form][c][dim]
              = dofmaps[form][dim]->cell_dofs(cell_index[c]);
            num_dofs[dim] += cell_dofs[form][c][dim].size();
          }

          // Resize macro dof holder
          for (std::size_t dim = 0; dim < rank; ++dim)
            macro_dofs[form][dim].resize(num_dofs[dim]);

          // Tabulate dofs on macro element
          const std::size_t rank = (form == 0) ? 2 : 1;
          for (std::size_t dim = 0; dim < rank; ++dim)
          {
            std::copy(cell_dofs[form][c][dim].begin(),
                      cell_dofs[form][c][dim].end(),
                      macro_dofs[form][dim].begin()
                      + c*cell_dofs[form][0][dim].size());
          }
        }


        // Get facet integral for sub domain (if any)
        if (use_interior_facet_domains)
        {
          const std::size_t domain = (*interior_facet_domains)[*facet];
          interior_facet_integrals[form]
            = ufc[form]->get_interior_facet_integral(domain);
        }

        // Check if facet tensor is required
        if (form == 0)
        {
          for (std::size_t c = 0; c < 2; ++c)
          {
            tensor_required_facet[form]
              = cell_matrix_required(tensors[form],
                                     interior_facet_integrals[form],
                                     boundary_values,
                                     cell_dofs[form][c][1]);
            if (tensor_required_facet[form])
              break;
          }
        }
        else
        {
          tensor_required_facet[form]
            = (tensors[form] && interior_facet_integrals[form]);
        }

        // Get cell integrals
        for (std::size_t c = 0; c < 2; ++c)
        {
          if (local_facet[c] == 0)
          {
            // Get cell integrals for sub domain (if any)
            if (use_cell_domains)
            {
              const std::size_t domain = (*cell_domains)[cell[c]];
              cell_integrals[form] = ufc[form]->get_cell_integral(domain);
            }

            // Check if facet tensor is required
            if (form == 0)
            {
              tensor_required_cell[form]
                = cell_matrix_required(tensors[form],
                                       cell_integrals[form],
                                       boundary_values,
                                       cell_dofs[form][c][1]);
            }
            else
              tensor_required_cell[form] = tensors[form] && cell_integrals[form];
          }
        }

        // Reset work array
        std::fill(ufc[form]->macro_A.begin(), ufc[form]->macro_A.end(), 0.0);
      }

      // Compute cell/facet tensor for lhs and rhs
      std::array<std::size_t, 2> matrix_size;
      std::size_t vector_size = 0;
      std::size_t cell_index = 0;
      for (std::size_t c = 0; c < num_cells; ++c)
      {
        if (local_facet[c] == 0)
        {
          matrix_size[0] = cell_dofs[0][c][0].size();
          matrix_size[1] = cell_dofs[0][c][1].size();
          vector_size = cell_dofs[1][c][0].size();
          cell_index = c;
        }
      }
      compute_interior_facet_tensor(ufc, ufc_cell,
                                    vertex_coordinates,
                                    tensor_required_cell,
                                    tensor_required_facet,
                                    cell, local_facet,
                                    facet_owner,
                                    cell_integrals,
                                    interior_facet_integrals,
                                    matrix_size,
                                    vector_size);

      // Modify local tensors for bcs
      ArrayView<const la_index> mdofs0(macro_dofs[0][0].size(),
                                       macro_dofs[0][0].data());
      ArrayView<const la_index> mdofs1(macro_dofs[0][1].size(),
                                       macro_dofs[0][1].data());
      apply_bc(ufc[0]->macro_A.data(), ufc[1]->macro_A.data(), boundary_values,
               mdofs0, mdofs1);

      // Add entries to global tensor
      if (tensors[1])
      {
        std::vector<ArrayView<const la_index>> mdofs(macro_dofs[1].size());
        for (std::size_t i = 0; i < macro_dofs[1].size(); ++i)
        {
          mdofs[i] = ArrayView<const la_index>(macro_dofs[1][i].size(),
                                               macro_dofs[1][i].data());
        }
        tensors[1]->add_local(ufc[1]->macro_A.data(), mdofs);
      }

      const bool add_macro_element
        = ufc[0]->form.has_interior_facet_integrals();
      if (tensors[0] && add_macro_element)
      {
        std::vector<ArrayView<const la_index>> mdofs(macro_dofs[0].size());
        for (std::size_t i = 0; i < macro_dofs[0].size(); ++i)
        {
          mdofs[i] = ArrayView<const la_index>(macro_dofs[0][i].size(),
                                               macro_dofs[0][i].data());
        }
        tensors[0]->add_local(ufc[0]->macro_A.data(), mdofs);
      }
      else if (tensors[0] && !add_macro_element)
      {
        // The sparsity pattern may not support the macro element so
        // instead extract back out the diagonal cell blocks and add
        // them individually
        matrix_block_add(*tensors[0], data.Ae[0], ufc[0]->macro_A,
                         tensor_required_cell[0], local_facet,
                         cell_dofs[0][cell_index]);
      }
    }
    else // Exterior facet
    {
      // Get mesh cell to which mesh facet belongs (pick first,
      // there is only one)
      Cell cell(mesh, facet->entities(mesh.topology().dim())[0]);

      // Decide if tensor needs to be computed
      for (std::size_t form = 0; form < 2; ++form)
      {
        // Get rank (lhs=2, rhs=1)
        const std::size_t rank = (form == 0) ? 2 : 1;

        // Get cell integrals for sub domain (if any)
        if (use_cell_domains)
        {
          const std::size_t domain = (*cell_domains)[cell];
          cell_integrals[form] = ufc[form]->get_cell_integral(domain);
        }

        // Get exterior facet integrals for sub domain (if any)
        if (use_exterior_facet_domains)
        {
          const std::size_t domain = (*exterior_facet_domains)[*facet];
          exterior_facet_integrals[form]
            = ufc[form]->get_exterior_facet_integral(domain);
        }

        // Get local-to-global dof maps for cell
        for (std::size_t dim = 0; dim < rank; ++dim)
        {
          cell_dofs[form][0][dim]
            = dofmaps[form][dim]->cell_dofs(cell.index());
        }

        // Store if tensor is required
        if (rank == 2)
        {
          tensor_required_facet[form]
            = cell_matrix_required(tensors[form],
                                   exterior_facet_integrals[form],
                                   boundary_values,
                                   cell_dofs[form][0][1]);
          tensor_required_cell[form]
            = cell_matrix_required(tensors[form],
                                   cell_integrals[form],
                                   boundary_values,
                                   cell_dofs[form][0][1]);
        }
        else
        {
          tensor_required_facet[form]
            = (tensors[form] && exterior_facet_integrals[form]);
          tensor_required_cell[form]
            = tensors[form] && cell_integrals[form];
        }
      }

      // Compute cell/facet tensors
      compute_exterior_facet_tensor(data.Ae, ufc, ufc_cell[0],
                                    vertex_coordinates[0],
                                    tensor_required_cell,
                                    tensor_required_facet,
                                    cell, *facet,
                                    cell_integrals,
                                    exterior_facet_integrals);

      // Modify local matrix/element for Dirichlet boundary conditions
      apply_bc(data.Ae[0].data(), data.Ae[1].data(), boundary_values,
               cell_dofs[0][0][0], cell_dofs[0][0][1]);

      // Add entries to global tensor
      for (std::size_t form = 0; form < 2; ++form)
      {
        if (tensors[form])
          tensors[form]->add_local(data.Ae[form].data(), cell_dofs[form][0]);
      }
    }
    p++;
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler:: compute_exterior_facet_tensor(
  std::array<std::vector<double>, 2>& Ae,
  std::array<UFC*, 2>& ufc,
  ufc::cell& ufc_cell,
  std::vector<double>& vertex_coordinates,
  const std::array<bool, 2>& tensor_required_cell,
  const std::array<bool, 2>& tensor_required_facet,
  const Cell& cell,
  const Facet& facet,
  const std::array<const ufc::cell_integral*, 2>& cell_integrals,
  const std::array<const ufc::exterior_facet_integral*, 2>& exterior_facet_integrals)
{
  // Get local index of facet with respect to the cell
  const std::size_t local_facet = cell.index(facet);

  // Get cell data
  cell.get_vertex_coordinates(vertex_coordinates);
  cell.get_cell_data(ufc_cell, local_facet);

  // Loop over lhs and then rhs facet contributions
  for (std::size_t form = 0; form < 2; ++form)
  {
    // Initialize macro element matrix/vector to zero
    std::fill(Ae[form].begin(), Ae[form].end(), 0.0);
    std::fill(ufc[form]->A.begin(), ufc[form]->A.end(), 0.0);

    // Compute facet integral,if required
    if (tensor_required_facet[form])
    {
      // Update UFC object
      ufc[form]->update(cell, vertex_coordinates, ufc_cell,
                        exterior_facet_integrals[form]->enabled_coefficients());
      exterior_facet_integrals[form]->tabulate_tensor(ufc[form]->A.data(),
                                                      ufc[form]->w(),
                                                      vertex_coordinates.data(),
                                                      local_facet,
                                                      ufc_cell.orientation);
      for (std::size_t i = 0; i < Ae[form].size(); i++)
        Ae[form][i] += ufc[form]->A[i];
    }

    // If we have local facet 0, assemble cell integral
    if (local_facet == 0)
    {
      dolfin_assert(!cell.is_ghost());

      // Compute cell integral, if required
      if (tensor_required_cell[form])
      {
        ufc[form]->update(cell, vertex_coordinates, ufc_cell,
                          cell_integrals[form]->enabled_coefficients());
        cell_integrals[form]->tabulate_tensor(ufc[form]->A.data(),
                                              ufc[form]->w(),
                                              vertex_coordinates.data(),
                                              ufc_cell.orientation);
        for (std::size_t i = 0; i < Ae[form].size(); i++)
          Ae[form][i] += ufc[form]->A[i];
      }
    }
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::compute_interior_facet_tensor(
  std::array<UFC*, 2>& ufc,
  std::array<ufc::cell, 2>& ufc_cell,
  std::array<std::vector<double>, 2>& vertex_coordinates,
  const std::array<bool, 2>& tensor_required_cell,
  const std::array<bool, 2>& tensor_required_facet,
  const std::array<Cell, 2>& cell,
  const std::array<std::size_t, 2>& local_facet,
  const bool facet_owner,
  const std::array<const ufc::cell_integral*, 2>& cell_integrals,
  const std::array<const ufc::interior_facet_integral*, 2>& interior_facet_integrals,
  const std::array<std::size_t, 2>& matrix_size,
  const std::size_t vector_size)
{
  // Compute facet contribution to tensor, if required
  // Loop over lhs and then rhs facet contributions
  for (std::size_t form = 0; form < 2; ++form)
  {
    // Compute interior facet integral
    if (tensor_required_facet[form] && facet_owner)
    {
      // Update to current pair of cells
      ufc[form]->update(cell[0], vertex_coordinates[0], ufc_cell[0],
                        cell[1], vertex_coordinates[1], ufc_cell[1],
                        interior_facet_integrals[form]->enabled_coefficients());
      // Integrate over facet
      interior_facet_integrals[form]->tabulate_tensor(ufc[form]->macro_A.data(),
                                                      ufc[form]->macro_w(),
                                                      vertex_coordinates[0].data(),
                                                      vertex_coordinates[1].data(),
                                                      local_facet[0],
                                                      local_facet[1],
                                                      ufc_cell[0].orientation,
                                                      ufc_cell[1].orientation);
    }

    // Compute cell contribution
    for (std::size_t c = 0; c < 2; ++c)
    {
      if (local_facet[c] == 0)
      {
        // Compute cell tensor, if required
        if (tensor_required_cell[form] and !cell[c].is_ghost())
        {
          ufc[form]->update(cell[c], vertex_coordinates[c], ufc_cell[c],
                            cell_integrals[form]->enabled_coefficients());
          cell_integrals[form]->tabulate_tensor(ufc[form]->A.data(),
                                                ufc[form]->w(),
                                                vertex_coordinates[c].data(),
                                                ufc_cell[c].orientation);

          // FIXME: Can the below two blocks be consolidated?
          const std::size_t nn = matrix_size[0];
          if (form == 0)
          {
            const std::size_t mm = matrix_size[1];
            for (std::size_t i = 0; i < mm; i++)
            {
              for (std::size_t j = 0; j < nn; j++)
              {
                ufc[form]->macro_A[2*nn*mm*c + 2*i*nn + nn*c + j]
                  += ufc[form]->A[i*nn + j];
              }
            }
          }
          else
          {
            for (std::size_t i = 0; i < vector_size; i++)
              ufc[form]->macro_A[nn*c + i] += ufc[form]->A[i];
          }
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::matrix_block_add(
  GenericTensor& tensor,
  std::vector<double>& Ae,
  std::vector<double>& macro_A,
  const bool tensor_required_cell,
  const std::array<std::size_t, 2>& local_facet,
  const std::vector<ArrayView<const la_index>>& cell_dofs)
{
  for (std::size_t c = 0; c < 2; ++c)
  {
    if (local_facet[c] == 0)
    {
      // Add cell tensor, if required
      if (tensor_required_cell)
      {
        std::fill(Ae.begin(), Ae.end(), 0.0);
        const std::size_t nn = cell_dofs[0].size();
        const std::size_t mm = cell_dofs[1].size();
        for (std::size_t i = 0; i < mm; i++)
        {
          for (std::size_t j = 0; j < nn; j++)
            Ae[i*nn + j] = macro_A[2*nn*mm*c + 2*i*nn + nn*c +j];
        }
        tensor.add_local(Ae.data(), cell_dofs);
      }
    }
  }
}
//-----------------------------------------------------------------------------
void
SystemAssembler::apply_bc(double* A, double* b,
                          const DirichletBC::Map& boundary_values,
                          const ArrayView<const dolfin::la_index>& global_dofs0,
                          const ArrayView<const dolfin::la_index>& global_dofs1)
{
  dolfin_assert(A);
  dolfin_assert(b);
  dolfin_assert(global_dofs0.size() == global_dofs1.size());

  // Wrap matrix and vector using Eigen
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>>
    _matA(A, global_dofs0.size(), global_dofs1.size());
  Eigen::Map<Eigen::VectorXd> _b(b, global_dofs1.size());

  // Loop over rows
  //for (std::size_t i = 0; i < _matA.n_rows; ++i)
  for (int i = 0; i < _matA.cols(); ++i)
  {
    const std::size_t ii = global_dofs1[i];
    DirichletBC::Map::const_iterator bc_value = boundary_values.find(ii);
    if (bc_value != boundary_values.end())
    {
      // Zero row
      _matA.row(i).setZero();

      // Modify RHS (subtract (bc_column(A))*bc_val from b)
      _b -= _matA.col(i)*bc_value->second;

      // Zero column
      _matA.col(i).setZero();

      // Place 1 on diagonal and bc on RHS (i th row ).
      _b(i)    = bc_value->second;
      _matA(i, i) = 1.0;
    }
  }
}
//-----------------------------------------------------------------------------
bool SystemAssembler::has_bc(const DirichletBC::Map& boundary_values,
                             const ArrayView<const dolfin::la_index>& dofs)
{
  // Loop over dofs and check if bc is applied
  for (auto dof = dofs.begin(); dof != dofs.end(); ++dof)
  {
    DirichletBC::Map::const_iterator bc_value = boundary_values.find(*dof);
    if (bc_value != boundary_values.end())
      return true;
  }

  return false;
}
//-----------------------------------------------------------------------------
bool SystemAssembler::cell_matrix_required(
  const GenericTensor* A,
  const void* integral,
  const DirichletBC::Map& boundary_values,
  const ArrayView<const dolfin::la_index>& dofs)
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
