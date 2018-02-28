// Copyright (C) 2008-2015 Kent-Andre Mardal and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SystemAssembler.h"
#include "AssemblerBase.h"
#include "DirichletBC.h"
#include "FiniteElement.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "UFC.h"
#include <algorithm>
#include <array>
#include <dolfin/common/ArrayView.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/types.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/function/GenericFunction.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/MeshIterator.h>
#include <dolfin/mesh/SubDomain.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
SystemAssembler::SystemAssembler(
    std::shared_ptr<const Form> a, std::shared_ptr<const Form> L,
    std::vector<std::shared_ptr<const DirichletBC>> bcs)
    : _a(a), _l(L), _bcs(bcs)
{
  // Check arity of forms
  check_arity(_a, _l);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(la::PETScMatrix& A, la::PETScVector& b)
{
  assemble(&A, &b, NULL);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(la::PETScMatrix& A) { assemble(&A, NULL, NULL); }
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(la::PETScVector& b) { assemble(NULL, &b, NULL); }
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(la::PETScMatrix& A, la::PETScVector& b,
                               const la::PETScVector& x0)
{
  assemble(&A, &b, &x0);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(la::PETScVector& b, const la::PETScVector& x0)
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
      dolfin_error("SystemAssembler.cpp", "assemble system",
                   "expected a bilinear form for a");
    }
  }

  // Check that L is a linear form
  if (L)
  {
    if (L->rank() != 1)
    {
      dolfin_error("SystemAssembler.cpp", "assemble system",
                   "expected a linear form for L");
    }
  }
}
//-----------------------------------------------------------------------------
std::shared_ptr<const mesh::MeshFunction<std::size_t>>
_pick_one_meshfunction(std::string name,
                       std::shared_ptr<const mesh::MeshFunction<std::size_t>> a,
                       std::shared_ptr<const mesh::MeshFunction<std::size_t>> b)
{
  if ((a && b) && a != b)
  {
    warning("Bilinear and linear forms do not have same %s subdomains in "
            "SystemAssembler. Taking %s subdomains from bilinear form",
            name.c_str(), name.c_str());
  }
  return a ? a : b;
}
//-----------------------------------------------------------------------------
bool SystemAssembler::check_functionspace_for_bc(
    std::shared_ptr<const function::FunctionSpace> fs, std::size_t bc_index)
{
  dolfin_assert(_bcs[bc_index]);
  std::shared_ptr<const function::FunctionSpace> bc_function_space
      = _bcs[bc_index]->function_space();
  dolfin_assert(bc_function_space);

  return fs->contains(*bc_function_space);
}
//-----------------------------------------------------------------------------
void SystemAssembler::assemble(la::PETScMatrix* A, la::PETScVector* b,
                               const la::PETScVector* x0)
{
  dolfin_assert(_a);
  dolfin_assert(_l);

  // Set timer
  common::Timer timer("Assemble system");

  // Get mesh
  dolfin_assert(_a->mesh());
  const mesh::Mesh& mesh = *(_a->mesh());
  dolfin_assert(mesh.ordered());

  // Get cell domains
  std::shared_ptr<const mesh::MeshFunction<std::size_t>> cell_domains
      = _pick_one_meshfunction("cell_domains", _a->cell_domains(),
                               _l->cell_domains());

  // Get exterior facet domains
  std::shared_ptr<const mesh::MeshFunction<std::size_t>> exterior_facet_domains
      = _pick_one_meshfunction("exterior_facet_domains",
                               _a->exterior_facet_domains(),
                               _l->exterior_facet_domains());

  // Get interior facet domains
  std::shared_ptr<const mesh::MeshFunction<std::size_t>> interior_facet_domains
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
  if (*_a->function_space(0) != *_l->function_space(0))
  {
    dolfin_error("SystemAssembler.cpp", "assemble system",
                 "expected forms (a, L) to share a function::FunctionSpace");
  }

  // Create data structures for local assembly data
  UFC A_ufc(*_a), b_ufc(*_l);

  // Raise error for Point integrals
  if (_a->integrals().num_vertex_integrals() > 0
      or _l->integrals().num_vertex_integrals() > 0)
  {
    dolfin_error("SystemAssembler.cpp", "assemble system",
                 "Point integrals are not supported (yet)");
  }

  // Gather UFC  objects
  std::array<UFC*, 2> ufc = {{&A_ufc, &b_ufc}};

  // Initialize global tensors
  if (A)
    init_global_tensor(*A, *_a);
  if (b)
    init_global_tensor(*b, *_l);

  // Gather tensors
  std::pair<la::PETScMatrix*, la::PETScVector*> tensors(A, b);

  // Allocate data
  Scratch data(*_a, *_l);

  // Get Dirichlet dofs and values for local mesh
  // Determine whether _a is bilinear in the same form
  bool rectangular = (*_a->function_space(0) != *_a->function_space(1));

  // Bin boundary conditions according to which form they apply to (if any)
  std::vector<DirichletBC::Map> boundary_values(rectangular ? 2 : 1);
  for (std::size_t i = 0; i < _bcs.size(); ++i)
  {
    // Match the function::FunctionSpace of the BC
    // with the (possible sub-)FunctionSpace on each axis of _a.
    bool axis0 = check_functionspace_for_bc(_a->function_space(0), i);
    bool axis1
        = rectangular && check_functionspace_for_bc(_a->function_space(1), i);

    // Fetch bc on axis0
    if (axis0)
    {
      log(TRACE, "System assembler: boundary condition %d applies to axis 0",
          i);
      _bcs[i]->get_boundary_values(boundary_values[0]);
      if (MPI::size(mesh.mpi_comm()) > 1 && _bcs[i]->method() != "pointwise")
        _bcs[i]->gather(boundary_values[0]);
    }

    // Fetch bc on axis1
    if (axis1)
    {
      log(TRACE, "System assembler: boundary condition %d applies to axis 1",
          i);
      _bcs[i]->get_boundary_values(boundary_values[1]);
      if (MPI::size(mesh.mpi_comm()) > 1 && _bcs[i]->method() != "pointwise")
        _bcs[i]->gather(boundary_values[1]);
    }

    if (!axis0 && !axis1)
    {
      log(TRACE,
          "System assembler: ignoring inapplicable boundary condition %d", i);
    }
  }

  // Modify boundary values for incremental (typically nonlinear)
  // problems
  // FIXME: not sure what happens when rectangular==true,
  //        should we raise "not implemented error" here?
  if (x0)
  {
    dolfin_assert(x0->size()
                  == _a->function_space(1)->dofmap()->global_dimension());

    const std::size_t num_bc_dofs = boundary_values[0].size();
    std::vector<dolfin::la_index_t> bc_indices;
    std::vector<double> bc_values;
    bc_indices.reserve(num_bc_dofs);
    bc_values.reserve(num_bc_dofs);

    // Build list of boundary dofs and values
    for (const auto& bv : boundary_values[0])
    {
      bc_indices.push_back(bv.first);
      bc_values.push_back(bv.second);
    }

    // Modify bc values
    std::vector<double> x0_values(num_bc_dofs);
    x0->get_local(x0_values.data(), num_bc_dofs, bc_indices.data());
    for (std::size_t i = 0; i < num_bc_dofs; i++)
      boundary_values[0][bc_indices[i]] = x0_values[i] - bc_values[i];
  }

  // Check whether we should do cell-wise or facet-wise assembly
  if (ufc[0]->dolfin_form.integrals().num_interior_facet_integrals() == 0
      && ufc[1]->dolfin_form.integrals().num_interior_facet_integrals() == 0)
  {
    // Assemble cell-wise (no interior facet integrals)
    cell_wise_assembly(tensors, ufc, data, boundary_values, cell_domains,
                       exterior_facet_domains);
  }
  else
  {
    // Assemble facet-wise (including cell assembly)
    facet_wise_assembly(tensors, ufc, data, boundary_values, cell_domains,
                        exterior_facet_domains, interior_facet_domains);
  }

  // Finalise assembly
  if (finalize_tensor)
  {
    if (A)
      A->apply(la::PETScMatrix::AssemblyType::FINAL);
    if (b)
      b->apply();
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::cell_wise_assembly(
    std::pair<la::PETScMatrix*, la::PETScVector*>& tensors, std::array<UFC*, 2>& ufc,
    Scratch& data, const std::vector<DirichletBC::Map>& boundary_values,
    std::shared_ptr<const mesh::MeshFunction<std::size_t>> cell_domains,
    std::shared_ptr<const mesh::MeshFunction<std::size_t>>
        exterior_facet_domains)
{
  // Extract mesh
  dolfin_assert(ufc[0]->dolfin_form.mesh());
  const mesh::Mesh& mesh = *(ufc[0]->dolfin_form.mesh());

  // Initialize entities if using external facet integrals
  dolfin_assert(mesh.ordered());
  bool has_exterior_facet_integrals
      = ufc[0]->dolfin_form.integrals().num_exterior_facet_integrals() > 0
        or ufc[1]->dolfin_form.integrals().num_exterior_facet_integrals() > 0;
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
  std::array<std::vector<common::ArrayView<const dolfin::la_index_t>>, 2>
      cell_dofs
      = {{std::vector<common::ArrayView<const dolfin::la_index_t>>(2),
          std::vector<common::ArrayView<const dolfin::la_index_t>>(1)}};

  // Create pointers to hold integral objects
  std::array<const ufc::cell_integral*, 2> cell_integrals
      = {{ufc[0]->dolfin_form.integrals().cell_integral().get(),
          ufc[1]->dolfin_form.integrals().cell_integral().get()}};

  std::array<const ufc::exterior_facet_integral*, 2> exterior_facet_integrals
      = {{ufc[0]->dolfin_form.integrals().exterior_facet_integral().get(),
          ufc[1]->dolfin_form.integrals().exterior_facet_integral().get()}};

  // Check whether integrals are domain-dependent
  bool use_cell_domains = cell_domains && !cell_domains->empty();
  bool use_exterior_facet_domains
      = exterior_facet_domains && !exterior_facet_domains->empty();

  la::PETScMatrix* A = tensors.first;
  la::PETScVector* b = tensors.second;

  // Iterate over all cells
  ufc::cell ufc_cell;
  EigenRowMatrixXd coordinate_dofs;
  std::size_t gdim = mesh.geometry().dim();

  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Check that cell is not a ghost
    dolfin_assert(!cell.is_ghost());

    // Get cell vertex coordinates
    coordinate_dofs.resize(cell.num_vertices(), gdim);
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get UFC cell data
    cell.get_cell_data(ufc_cell);

    // Loop over lhs and then rhs contributions
    for (std::size_t form = 0; form < 2; ++form)
    {
      // Don't need to assemble rhs if only system matrix is required
      if (form == 1 && !b)
        continue;

      // Get rank (lhs=2, rhs=1)
      const std::size_t rank = (form == 0) ? 2 : 1;

      // Zero data
      std::fill(data.Ae[form].begin(), data.Ae[form].end(), 0.0);

      // Get cell integrals for sub domain (if any)
      if (use_cell_domains)
      {
        const std::size_t domain = (*cell_domains)[cell];
        cell_integrals[form]
            = ufc[form]->dolfin_form.integrals().cell_integral(domain).get();
      }

      // Get local-to-global dof maps for cell
      for (std::size_t dim = 0; dim < rank; ++dim)
      {
        auto dmap = dofmaps[form][dim]->cell_dofs(cell.index());
        cell_dofs[form][dim].set(dmap.size(), dmap.data());
      }

      // Compute cell tensor (if required)
      bool tensor_required;
      if (rank == 2) // form == 0
      {
        tensor_required = cell_matrix_required(
            A, cell_integrals[form], boundary_values, cell_dofs[form][1]);
      }
      else
        tensor_required = b && cell_integrals[form];

      if (tensor_required)
      {
        // Update to current cell
        ufc[form]->update(cell, coordinate_dofs, ufc_cell,
                          cell_integrals[form]->enabled_coefficients());

        // Tabulate cell tensor
        cell_integrals[form]->tabulate_tensor(
            ufc[form]->A.data(), ufc[form]->w(), coordinate_dofs.data(),
            ufc_cell.orientation);
        for (std::size_t i = 0; i < data.Ae[form].size(); ++i)
          data.Ae[form][i] += ufc[form]->A[i];
      }

      // Compute exterior facet integral if present
      if (has_exterior_facet_integrals)
      {
        for (auto& facet : mesh::EntityRange<mesh::Facet>(cell))
        {
          // Only consider exterior facets
          if (!facet.exterior())
            continue;

          // Get exterior facet integrals for sub domain (if any)
          if (use_exterior_facet_domains)
          {
            const std::size_t domain = (*exterior_facet_domains)[facet];
            exterior_facet_integrals[form]
                = ufc[form]
                      ->dolfin_form.integrals()
                      .exterior_facet_integral(domain)
                      .get();
          }

          // Skip if there are no integrals
          if (!exterior_facet_integrals[form])
            continue;

          // Extract local facet index
          const std::size_t local_facet = cell.index(facet);

          // Determine if tensor needs to be computed
          bool tensor_required;
          if (rank == 2) // form == 0
          {
            tensor_required
                = cell_matrix_required(A, exterior_facet_integrals[form],
                                       boundary_values, cell_dofs[form][1]);
          }
          else
            tensor_required = b;

          // Add exterior facet tensor
          if (tensor_required)
          {
            // Update to current cell
            cell.get_cell_data(ufc_cell);
            ufc[form]->update(
                cell, coordinate_dofs, ufc_cell,
                exterior_facet_integrals[form]->enabled_coefficients());

            // Tabulate exterior facet tensor
            exterior_facet_integrals[form]->tabulate_tensor(
                ufc[form]->A.data(), ufc[form]->w(), coordinate_dofs.data(),
                local_facet, ufc_cell.orientation);
            for (std::size_t i = 0; i < data.Ae[form].size(); i++)
              data.Ae[form][i] += ufc[form]->A[i];
          }
        }
      }
    }

    // Modify local matrix/element for Dirichlet boundary conditions
    apply_bc(data.Ae[0].data(), data.Ae[1].data(), boundary_values,
             cell_dofs[0][0], cell_dofs[0][1]);

    // Add entries to global tensor
    if (A)
      A->add_local(data.Ae[0].data(), cell_dofs[0][0].size(),
                   cell_dofs[0][0].data(), cell_dofs[0][1].size(),
                   cell_dofs[0][1].data());
    if (b)
      b->add_local(data.Ae[1].data(), cell_dofs[1][0].size(),
                   cell_dofs[1][0].data());
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::facet_wise_assembly(
    std::pair<la::PETScMatrix*, la::PETScVector*>& tensors, std::array<UFC*, 2>& ufc,
    Scratch& data, const std::vector<DirichletBC::Map>& boundary_values,
    std::shared_ptr<const mesh::MeshFunction<std::size_t>> cell_domains,
    std::shared_ptr<const mesh::MeshFunction<std::size_t>>
        exterior_facet_domains,
    std::shared_ptr<const mesh::MeshFunction<std::size_t>>
        interior_facet_domains)
{
  // Extract mesh
  dolfin_assert(ufc[0]->dolfin_form.mesh());
  const mesh::Mesh& mesh = *(ufc[0]->dolfin_form.mesh());

  // Sanity check of ghost mode (proper check in AssemblerBase::check)
  dolfin_assert(mesh.ghost_mode() == "shared_vertex"
                || mesh.ghost_mode() == "shared_facet"
                || MPI::size(mesh.mpi_comm()) == 1);

  // Compute facets and facet - cell connectivity if not already
  // computed
  const std::size_t D = mesh.topology().dim();
  mesh.init(D - 1);
  mesh.init(D - 1, D);

  // Get my MPI rank
  const int my_mpi_rank = MPI::rank(mesh.mpi_comm());

  // Collect pointers to dof maps
  std::array<std::vector<const GenericDofMap*>, 2> dofmaps;
  for (std::size_t i = 0; i < 2; ++i)
    dofmaps[0].push_back(ufc[0]->dolfin_form.function_space(i)->dofmap().get());
  dofmaps[1].push_back(ufc[1]->dolfin_form.function_space(0)->dofmap().get());

  // Cell dofmaps [form][cell][form dim]
  std::
      array<std::array<std::vector<common::ArrayView<const dolfin::la_index_t>>,
                       2>,
            2>
          cell_dofs;
  cell_dofs[0][0].resize(2);
  cell_dofs[0][1].resize(2);
  cell_dofs[1][0].resize(1);
  cell_dofs[1][1].resize(1);

  std::array<mesh::Cell, 2> cell;
  std::array<std::size_t, 2> cell_index;
  std::array<std::size_t, 2> local_facet;

  // Vectors to hold dofs for macro cells
  std::array<std::vector<std::vector<dolfin::la_index_t>>, 2> macro_dofs;
  macro_dofs[0].resize(2);
  macro_dofs[1].resize(1);

  // Holder for number of dofs in macro-dofmap
  std::vector<std::size_t> num_dofs(2);

  // Holders for UFC integrals
  std::array<const ufc::cell_integral*, 2> cell_integrals
      = {{ufc[0]->dolfin_form.integrals().cell_integral().get(),
          ufc[1]->dolfin_form.integrals().cell_integral().get()}};
  std::array<const ufc::exterior_facet_integral*, 2> exterior_facet_integrals
      = {{ufc[0]->dolfin_form.integrals().exterior_facet_integral().get(),
          ufc[1]->dolfin_form.integrals().exterior_facet_integral().get()}};
  std::array<const ufc::interior_facet_integral*, 2> interior_facet_integrals
      = {{ufc[0]->dolfin_form.integrals().interior_facet_integral().get(),
          ufc[1]->dolfin_form.integrals().interior_facet_integral().get()}};

  // Check whether integrals are domain-dependent
  bool use_cell_domains = cell_domains && !cell_domains->empty();
  bool use_interior_facet_domains
      = interior_facet_domains && !interior_facet_domains->empty();
  bool use_exterior_facet_domains
      = exterior_facet_domains && !exterior_facet_domains->empty();

  // Indicator whether or not tensor is required
  std::array<bool, 2> tensor_required_cell = {{false, false}};
  std::array<bool, 2> tensor_required_facet = {{false, false}};

  // Track whether or not cell contribution has been computed
  std::array<bool, 2> compute_cell_tensor = {{true, true}};
  std::vector<bool> cell_tensor_computed(mesh.num_cells(), false);

  la::PETScMatrix* A = tensors.first;
  la::PETScVector* b = tensors.second;

  // Iterate over facets
  std::array<ufc::cell, 2> ufc_cell;
  std::array<EigenRowMatrixXd, 2> coordinate_dofs;
  const std::size_t gdim = mesh.geometry().dim();

  for (auto& facet : mesh::MeshRange<mesh::Facet>(mesh))
  {
    // Number of cells sharing facet
    const std::size_t num_cells = facet.num_entities(D);

    // Check that facet is not a ghost
    dolfin_assert(!facet.is_ghost());

    // Interior facet
    if (num_cells == 2)
    {
      // Get cells incident with facet (which is 0 and 1 here is arbitrary)
      dolfin_assert(facet.num_entities(D) == 2);
      std::array<std::int32_t, 2> cell_indices
          = {{facet.entities(D)[0], facet.entities(D)[1]}};

      // Make sure cell marker for '+' side is larger than cell marker
      // for '-' side.  Note: by ffc convention, 0 is + and 1 is -
      if (use_cell_domains
          && (*cell_domains)[cell_indices[0]]
                 < (*cell_domains)[cell_indices[1]])
      {
        std::swap(cell_indices[0], cell_indices[1]);
      }

      // Get cells incident with facet and associated data
      for (std::size_t c = 0; c < 2; ++c)
      {
        cell[c] = mesh::Cell(mesh, cell_indices[c]);
        cell_index[c] = cell[c].index();
        local_facet[c] = cell[c].index(facet);
        coordinate_dofs[c].resize(cell[c].num_vertices(), gdim);
        cell[c].get_coordinate_dofs(coordinate_dofs[c]);
        cell[c].get_cell_data(ufc_cell[c], local_facet[c]);

        compute_cell_tensor[c] = !cell_tensor_computed[cell_index[c]];
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
        // Don't need to assemble rhs if only system matrix is required
        if (form == 1 && !b)
          continue;

        // Get rank (lhs=2, rhs=1)
        const std::size_t rank = (form == 0) ? 2 : 1;

        // Compute number of dofs in macro dofmap
        std::fill(num_dofs.begin(), num_dofs.begin() + rank, 0);
        for (std::size_t c = 0; c < 2; ++c)
        {
          for (std::size_t dim = 0; dim < rank; ++dim)
          {
            auto dmap = dofmaps[form][dim]->cell_dofs(cell_index[c]);
            cell_dofs[form][c][dim].set(dmap.size(), dmap.data());
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
                          + c * cell_dofs[form][0][dim].size());
          }
        }

        // Get facet integral for sub domain (if any)
        if (use_interior_facet_domains)
        {
          const std::size_t domain = (*interior_facet_domains)[facet];
          interior_facet_integrals[form] = ufc[form]
                                               ->dolfin_form.integrals()
                                               .interior_facet_integral(domain)
                                               .get();
        }

        // Check if facet tensor is required
        if (rank == 2)
        {
          for (std::size_t c = 0; c < 2; ++c)
          {
            tensor_required_facet[form]
                = cell_matrix_required(A, interior_facet_integrals[form],
                                       boundary_values, cell_dofs[form][c][1]);
            if (tensor_required_facet[form])
              break;
          }
        }
        else
        {
          tensor_required_facet[form] = (b && interior_facet_integrals[form]);
        }

        // Get cell integrals (if required)
        for (std::size_t c = 0; c < 2; ++c)
        {
          if (compute_cell_tensor[c])
          {
            // Get cell integrals for sub domain (if any)
            if (use_cell_domains)
            {
              const std::size_t domain = (*cell_domains)[cell[c]];
              cell_integrals[form] = ufc[form]
                                         ->dolfin_form.integrals()
                                         .cell_integral(domain)
                                         .get();
            }

            // Check if facet tensor is required
            if (form == 0)
            {
              tensor_required_cell[form] = cell_matrix_required(
                  A, cell_integrals[form], boundary_values,
                  cell_dofs[form][c][1]);
            }
            else
              tensor_required_cell[form] = b && cell_integrals[form];
          }
        }

        // Reset work array
        std::fill(ufc[form]->macro_A.begin(), ufc[form]->macro_A.end(), 0.0);
      }

      // Compute cell/facet tensor for lhs and rhs
      std::array<std::size_t, 2> matrix_size;
      std::size_t vector_size = 0;
      for (std::size_t c = 0; c < 2; ++c)
      {
        matrix_size[0] = cell_dofs[0][c][0].size();
        matrix_size[1] = cell_dofs[0][c][1].size();
        vector_size = cell_dofs[1][c][0].size();
      }
      compute_interior_facet_tensor(
          ufc, ufc_cell, coordinate_dofs, tensor_required_cell,
          tensor_required_facet, cell, local_facet, facet_owner, cell_integrals,
          interior_facet_integrals, matrix_size, vector_size,
          compute_cell_tensor);

      // Modify local tensors for bcs
      common::ArrayView<const la_index_t> mdofs0(macro_dofs[0][0]);
      common::ArrayView<const la_index_t> mdofs1(macro_dofs[0][1]);
      apply_bc(ufc[0]->macro_A.data(), ufc[1]->macro_A.data(), boundary_values,
               mdofs0, mdofs1);

      // Add entries to global tensor
      if (b)
      {
        std::vector<common::ArrayView<const la_index_t>> mdofs(
            macro_dofs[1].size());
        for (std::size_t i = 0; i < macro_dofs[1].size(); ++i)
          mdofs[i].set(macro_dofs[1][i]);
        b->add_local(ufc[1]->macro_A.data(), mdofs[0].size(), mdofs[0].data());
      }

      const bool add_macro_element
          = ufc[0]->dolfin_form.integrals().num_interior_facet_integrals() > 0;
      if (A && add_macro_element)
      {
        std::vector<common::ArrayView<const la_index_t>> mdofs(
            macro_dofs[0].size());
        for (std::size_t i = 0; i < macro_dofs[0].size(); ++i)
          mdofs[i].set(macro_dofs[0][i]);
        A->add_local(ufc[0]->macro_A.data(), mdofs[0].size(), mdofs[0].data(),
                     mdofs[1].size(), mdofs[1].data());
      }
      else if (A && !add_macro_element && tensor_required_cell[0])
      {
        // FIXME: This can be simplied by assembling into Ae instead
        // of macro_A.

        // The sparsity pattern may not support the macro element so
        // instead extract back out the diagonal cell blocks and add
        // them individually
        matrix_block_add(*A, data.Ae[0], ufc[0]->macro_A, compute_cell_tensor,
                         cell_dofs[0]);
      }

      // Mark cells as processed
      cell_tensor_computed[cell_index[0]] = true;
      cell_tensor_computed[cell_index[1]] = true;
    }
    else // Exterior facet
    {
      // Get mesh cell to which mesh facet belongs (pick first, there
      // is only one)
      mesh::Cell cell(mesh, facet.entities(mesh.topology().dim())[0]);

      // Check of attached cell needs to be processed
      compute_cell_tensor[0] = !cell_tensor_computed[cell.index()];

      // Decide if tensor needs to be computed
      for (std::size_t form = 0; form < 2; ++form)
      {
        // Get rank (lhs=2, rhs=1)
        const std::size_t rank = (form == 0) ? 2 : 1;

        // Get cell integrals for sub domain (if any)
        if (use_cell_domains)
        {
          const std::size_t domain = (*cell_domains)[cell];
          cell_integrals[form]
              = ufc[form]->dolfin_form.integrals().cell_integral(domain).get();
        }

        // Get exterior facet integrals for sub domain (if any)
        if (use_exterior_facet_domains)
        {
          const std::size_t domain = (*exterior_facet_domains)[facet];
          exterior_facet_integrals[form] = ufc[form]
                                               ->dolfin_form.integrals()
                                               .exterior_facet_integral(domain)
                                               .get();
        }

        // Get local-to-global dof maps for cell
        for (std::size_t dim = 0; dim < rank; ++dim)
        {
          auto dmap = dofmaps[form][dim]->cell_dofs(cell.index());
          cell_dofs[form][0][dim].set(dmap.size(), dmap.data());
        }

        // Store if tensor is required
        if (rank == 2)
        {
          tensor_required_facet[form]
              = cell_matrix_required(A, exterior_facet_integrals[form],
                                     boundary_values, cell_dofs[form][0][1]);
          tensor_required_cell[form] = cell_matrix_required(
              A, cell_integrals[form], boundary_values, cell_dofs[form][0][1]);
        }
        else
        {
          tensor_required_facet[form] = (b && exterior_facet_integrals[form]);
          tensor_required_cell[form] = b && cell_integrals[form];
        }
      }

      // Compute cell/facet tensors
      coordinate_dofs[0].resize(cell.num_vertices(), gdim);
      compute_exterior_facet_tensor(
          data.Ae, ufc, ufc_cell[0], coordinate_dofs[0], tensor_required_cell,
          tensor_required_facet, cell, facet, cell_integrals,
          exterior_facet_integrals, compute_cell_tensor[0]);

      // Modify local matrix/element for Dirichlet boundary conditions
      apply_bc(data.Ae[0].data(), data.Ae[1].data(), boundary_values,
               cell_dofs[0][0][0], cell_dofs[0][0][1]);

      // Add entries to global tensor
      if (A)
      {
        A->add_local(data.Ae[0].data(), cell_dofs[0][0][0].size(),
                     cell_dofs[0][0][0].data(), cell_dofs[0][0][1].size(),
                     cell_dofs[0][0][1].data());
      }

      if (b)
      {
        b->add_local(data.Ae[1].data(), cell_dofs[1][0][0].size(),
                     cell_dofs[1][0][0].data());
      }

      // Mark cell as processed
      cell_tensor_computed[cell.index()] = true;
    }
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::compute_exterior_facet_tensor(
    std::array<std::vector<double>, 2>& Ae, std::array<UFC*, 2>& ufc,
    ufc::cell& ufc_cell, Eigen::Ref<EigenRowMatrixXd> coordinate_dofs,
    const std::array<bool, 2>& tensor_required_cell,
    const std::array<bool, 2>& tensor_required_facet, const mesh::Cell& cell,
    const mesh::Facet& facet,
    const std::array<const ufc::cell_integral*, 2>& cell_integrals,
    const std::array<const ufc::exterior_facet_integral*, 2>&
        exterior_facet_integrals,
    const bool compute_cell_tensor)
{
  // Get local index of facet with respect to the cell
  const std::size_t local_facet = cell.index(facet);

  // Get cell data
  cell.get_coordinate_dofs(coordinate_dofs);
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
      ufc[form]->update(cell, coordinate_dofs, ufc_cell,
                        exterior_facet_integrals[form]->enabled_coefficients());
      exterior_facet_integrals[form]->tabulate_tensor(
          ufc[form]->A.data(), ufc[form]->w(), coordinate_dofs.data(),
          local_facet, ufc_cell.orientation);
      for (std::size_t i = 0; i < Ae[form].size(); i++)
        Ae[form][i] += ufc[form]->A[i];
    }

    // Assemble cell integral (if required)
    if (compute_cell_tensor)
    {
      dolfin_assert(!cell.is_ghost());

      // Compute cell integral, if required
      if (tensor_required_cell[form])
      {
        ufc[form]->update(cell, coordinate_dofs, ufc_cell,
                          cell_integrals[form]->enabled_coefficients());
        cell_integrals[form]->tabulate_tensor(
            ufc[form]->A.data(), ufc[form]->w(), coordinate_dofs.data(),
            ufc_cell.orientation);
        for (std::size_t i = 0; i < Ae[form].size(); i++)
          Ae[form][i] += ufc[form]->A[i];
      }
    }
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::compute_interior_facet_tensor(
    std::array<UFC*, 2>& ufc, std::array<ufc::cell, 2>& ufc_cell,
    std::array<EigenRowMatrixXd, 2>& coordinate_dofs,
    const std::array<bool, 2>& tensor_required_cell,
    const std::array<bool, 2>& tensor_required_facet,
    const std::array<mesh::Cell, 2>& cell,
    const std::array<std::size_t, 2>& local_facet, const bool facet_owner,
    const std::array<const ufc::cell_integral*, 2>& cell_integrals,
    const std::array<const ufc::interior_facet_integral*, 2>&
        interior_facet_integrals,
    const std::array<std::size_t, 2>& matrix_size,
    const std::size_t vector_size,
    const std::array<bool, 2> compute_cell_tensor)
{
  // Compute facet contribution to tensor, if required
  // Loop over lhs and then rhs facet contributions
  for (std::size_t form = 0; form < 2; ++form)
  {
    // Compute interior facet integral
    if (tensor_required_facet[form] && facet_owner)
    {
      // Update to current pair of cells
      ufc[form]->update(cell[0], coordinate_dofs[0], ufc_cell[0], cell[1],
                        coordinate_dofs[1], ufc_cell[1],
                        interior_facet_integrals[form]->enabled_coefficients());
      // Integrate over facet
      interior_facet_integrals[form]->tabulate_tensor(
          ufc[form]->macro_A.data(), ufc[form]->macro_w(),
          coordinate_dofs[0].data(), coordinate_dofs[1].data(), local_facet[0],
          local_facet[1], ufc_cell[0].orientation, ufc_cell[1].orientation);
    }

    // Compute cell contribution
    for (std::size_t c = 0; c < 2; ++c)
    {
      if (compute_cell_tensor[c])
      {
        // Compute cell tensor, if required
        if (tensor_required_cell[form] and !cell[c].is_ghost())
        {
          ufc[form]->update(cell[c], coordinate_dofs[c], ufc_cell[c],
                            cell_integrals[form]->enabled_coefficients());
          cell_integrals[form]->tabulate_tensor(
              ufc[form]->A.data(), ufc[form]->w(), coordinate_dofs[c].data(),
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
                ufc[form]->macro_A[2 * nn * mm * c + 2 * i * nn + nn * c + j]
                    += ufc[form]->A[i * nn + j];
              }
            }
          }
          else
          {
            for (std::size_t i = 0; i < vector_size; i++)
              ufc[form]->macro_A[nn * c + i] += ufc[form]->A[i];
          }
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::matrix_block_add(
    la::PETScMatrix& tensor, std::vector<double>& Ae, std::vector<double>& macro_A,
    const std::array<bool, 2>& add_local_tensor,
    const std::array<std::vector<common::ArrayView<const la_index_t>>, 2>&
        cell_dofs)
{
  for (std::size_t c = 0; c < 2; ++c)
  {
    // Add cell tensor, if not already processed
    if (add_local_tensor[c])
    {
      std::fill(Ae.begin(), Ae.end(), 0.0);
      const std::size_t nn = cell_dofs[c][0].size();
      const std::size_t mm = cell_dofs[c][1].size();
      for (std::size_t i = 0; i < mm; i++)
      {
        for (std::size_t j = 0; j < nn; j++)
          Ae[i * nn + j] = macro_A[2 * nn * mm * c + 2 * i * nn + nn * c + j];
      }

      tensor.add_local(Ae.data(), cell_dofs[c][0].size(),
                       cell_dofs[c][0].data(), cell_dofs[c][1].size(),
                       cell_dofs[c][1].data());
    }
  }
}
//-----------------------------------------------------------------------------
void SystemAssembler::apply_bc(
    double* A, double* b, const std::vector<DirichletBC::Map>& boundary_values,
    const common::ArrayView<const dolfin::la_index_t>& global_dofs0,
    const common::ArrayView<const dolfin::la_index_t>& global_dofs1)
{
  dolfin_assert(A);
  dolfin_assert(b);

  // Wrap matrix and vector using Eigen
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>>
      _matA(A, global_dofs0.size(), global_dofs1.size());
  Eigen::Map<Eigen::VectorXd> _b(b, global_dofs0.size());

  if (boundary_values.size() == 1)
  {
    // Square matrix with same function::FunctionSpace on each axis
    // Loop over columns/rows
    for (int i = 0; i < _matA.cols(); ++i)
    {
      const std::size_t ii = global_dofs1[i];
      DirichletBC::Map::const_iterator bc_value = boundary_values[0].find(ii);
      if (bc_value != boundary_values[0].end())
      {
        // Zero row
        _matA.row(i).setZero();

        // Modify RHS (subtract (bc_column(A))*bc_val from b)
        _b -= _matA.col(i) * bc_value->second;

        // Zero column
        _matA.col(i).setZero();

        // Place 1 on diagonal and bc on RHS (i th row ).
        _b(i) = bc_value->second;
        _matA(i, i) = 1.0;
      }
    }
  }
  else
  {
    // Possibly rectangular matrix with different spaces on axes
    // FIXME: This won't work for forms on W x W.sub(0), which would contain
    //        diagonal. This is difficult to distinguish from form
    //        on V x Q which would by accident contain V bc and Q bc with
    //        same dof index, but that's not diagonal.
    //
    //        Essentially we need to detect if _a->function_space(0) and
    //        _a->function_space(1) share the common super space (use
    //        function::FunctionSpace::_root_space_id). In that case dof ids are
    //        shared
    //        and matrix has diagonal. Otherwise it does not have diagonal.

    // Loop over rows first
    for (int i = 0; i < _matA.rows(); ++i)
    {
      const std::size_t ii = global_dofs0[i];
      DirichletBC::Map::const_iterator bc_value = boundary_values[0].find(ii);
      if (bc_value != boundary_values[0].end())
        _matA.row(i).setZero();
    }

    // Loop over columns
    for (int j = 0; j < _matA.cols(); ++j)
    {
      const std::size_t jj = global_dofs1[j];
      DirichletBC::Map::const_iterator bc_value = boundary_values[1].find(jj);
      if (bc_value != boundary_values[1].end())
      {
        // Modify RHS (subtract (bc_column(A))*bc_val from b)
        _b -= _matA.col(j) * bc_value->second;
        _matA.col(j).setZero();
      }
    }
  }
}
//-----------------------------------------------------------------------------
bool SystemAssembler::has_bc(
    const DirichletBC::Map& boundary_values,
    const common::ArrayView<const dolfin::la_index_t>& dofs)
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
    const la::PETScMatrix* A, const void* integral,
    const std::vector<DirichletBC::Map>& boundary_values,
    const common::ArrayView<const dolfin::la_index_t>& dofs)
{
  if (A && integral)
    return true;
  else if (integral && has_bc(boundary_values[0], dofs))
    return true;
  else
    return false;
}
//-----------------------------------------------------------------------------
SystemAssembler::Scratch::Scratch(const Form& a, const Form& L)
{
  std::size_t A_num_entries = a.function_space(0)->dofmap()->max_element_dofs();
  A_num_entries *= a.function_space(1)->dofmap()->max_element_dofs();
  Ae[0].resize(A_num_entries);
  Ae[1].resize(L.function_space(0)->dofmap()->max_element_dofs());
}
//-----------------------------------------------------------------------------
SystemAssembler::Scratch::~Scratch()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
