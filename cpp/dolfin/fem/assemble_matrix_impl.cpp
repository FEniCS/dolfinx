// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_matrix_impl.h"
#include "Form.h"
#include "GenericDofMap.h"
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/utils.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <petscsys.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void fem::impl::assemble_matrix(Mat A, const Form& a,
                                const std::vector<bool>& bc0,
                                const std::vector<bool>& bc1)
{
  assert(a.mesh());
  const mesh::Mesh& mesh = *a.mesh();

  // Get dofmap data
  const fem::GenericDofMap& dofmap0 = *a.function_space(0)->dofmap();
  const fem::GenericDofMap& dofmap1 = *a.function_space(1)->dofmap();
  Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dof_array0
      = dofmap0.dof_array();
  Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dof_array1
      = dofmap1.dof_array();
  // FIXME: do this right
  const int num_dofs_per_cell0 = dofmap0.num_element_dofs(0);
  const int num_dofs_per_cell1 = dofmap1.num_element_dofs(0);

  // Prepare coefficients
  const FormCoefficients& coefficients = a.coeffs();
  std::vector<const function::Function*> coeff_fn(coefficients.size());
  for (int i = 0; i < coefficients.size(); ++i)
    coeff_fn[i] = coefficients.get(i).get();
  std::vector<int> c_offsets = coefficients.offsets();

  const FormIntegrals& integrals = a.integrals();
  using type = fem::FormIntegrals::Type;
  for (int i = 0; i < integrals.num_integrals(type::cell); ++i)
  {
    auto& fn = integrals.get_tabulate_tensor_fn_cell(i);
    const std::vector<std::int32_t>& active_cells
        = integrals.integral_domains(type::cell, i);
    fem::impl::assemble_cells(
        A, mesh, active_cells, dof_array0, num_dofs_per_cell0, dof_array1,
        num_dofs_per_cell1, bc0, bc1, fn, coeff_fn, c_offsets);
  }

  for (int i = 0; i < integrals.num_integrals(type::exterior_facet); ++i)
  {
    auto& fn = integrals.get_tabulate_tensor_fn_exterior_facet(i);
    const std::vector<std::int32_t>& active_facets
        = integrals.integral_domains(type::exterior_facet, i);
    fem::impl::assemble_exterior_facets(A, mesh, active_facets, dofmap0,
                                        dofmap1, bc0, bc1, fn, coeff_fn,
                                        c_offsets);
  }

  if (a.integrals().num_integrals(type::interior_facet) > 0)
  {
    throw std::runtime_error(
        "Interior facet integrals in bilinear forms not yet supported.");
  }
}
//-----------------------------------------------------------------------------
void fem::impl::assemble_cells(
    Mat A, const mesh::Mesh& mesh,
    const std::vector<std::int32_t>& active_cells,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dofmap0,
    int num_dofs_per_cell0,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dofmap1,
    int num_dofs_per_cell1, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1,
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int)>& kernel,
    std::vector<const function::Function*> coefficients,
    const std::vector<int>& offsets)
{
  assert(A);

  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();

  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = mesh.coordinate_dofs().entity_points(tdim);
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> pos_g
      = connectivity_g.entity_positions();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
      x_g
      = mesh.geometry().points();

  // Data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(offsets.back());

  // Iterate over active cells
  PetscErrorCode ierr;
  for (auto& cell_index : active_cells)
  {
    const mesh::Cell cell(mesh, cell_index);

    // Check that cell is not a ghost
    assert(!cell.is_ghost());

    // Get cell coordinates/geometry
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Update coefficients
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
      coefficients[i]->restrict(coeff_array.data() + offsets[i], cell,
                                coordinate_dofs);
    }

    // Tabulate tensor
    Ae.setZero(num_dofs_per_cell0, num_dofs_per_cell1);
    kernel(Ae.data(), coeff_array.data(), coordinate_dofs.data(), 1);

    // Zero rows/columns for essential bcs
    if (!bc0.empty())
    {
      for (Eigen::Index i = 0; i < Ae.rows(); ++i)
      {
        const PetscInt dof = dofmap0[cell_index * num_dofs_per_cell0 + i];
        if (bc0[dof])
          Ae.row(i).setZero();
      }
    }
    if (!bc1.empty())
    {
      for (Eigen::Index j = 0; j < Ae.cols(); ++j)
      {
        const PetscInt dof = dofmap1[cell_index * num_dofs_per_cell1 + j];
        if (bc1[dof])
          Ae.col(j).setZero();
      }
    }

    ierr = MatSetValuesLocal(
        A, num_dofs_per_cell0, dofmap0.data() + cell_index * num_dofs_per_cell0,
        num_dofs_per_cell1, dofmap1.data() + cell_index * num_dofs_per_cell1,
        Ae.data(), ADD_VALUES);
#ifdef DEBUG
    if (ierr != 0)
      la::petsc_error(ierr, __FILE__, "MatSetValuesLocal");
#endif
  }
}
//-----------------------------------------------------------------------------
void fem::impl::assemble_exterior_facets(
    Mat A, const mesh::Mesh& mesh,
    const std::vector<std::int32_t>& active_facets,
    const GenericDofMap& dofmap0, const GenericDofMap& dofmap1,
    const std::vector<bool>& bc0, const std::vector<bool>& bc1,
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int, int)>& fn,
    std::vector<const function::Function*> coefficients,
    const std::vector<int>& offsets)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();
  mesh.create_entities(tdim - 1);
  mesh.create_connectivity(tdim - 1, tdim);

  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = mesh.coordinate_dofs().entity_points(tdim);
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> pos_g
      = connectivity_g.entity_positions();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
      x_g
      = mesh.geometry().points();

  // Data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(offsets.back());

  // Iterate over all facets
  PetscErrorCode ierr;
  for (const auto& facet_index : active_facets)
  {
    const mesh::Facet facet(mesh, facet_index);
    assert(facet.num_global_entities(tdim) == 1);

    // TODO: check ghosting sanity?

    // Create attached cell
    const mesh::Cell cell(mesh, facet.entities(tdim)[0]);

    // Get local index of facet with respect to the cell
    const int local_facet = cell.index(facet);

    // Get cell vertex coordinates
    const int cell_index = cell.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Get dof maps for cell
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap0
        = dofmap0.cell_dofs(cell_index);
    Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap1
        = dofmap1.cell_dofs(cell_index);

    // Update coefficients
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
      coefficients[i]->restrict(coeff_array.data() + offsets[i], cell,
                                coordinate_dofs);
    }

    // Tabulate tensor
    Ae.setZero(dmap0.size(), dmap1.size());
    fn(Ae.data(), coeff_array.data(), coordinate_dofs.data(), local_facet, 1);

    // Zero rows/columns for essential bcs
    if (!bc0.empty())
    {
      for (Eigen::Index i = 0; i < Ae.rows(); ++i)
      {
        if (bc0[dmap0[i]])
          Ae.row(i).setZero();
      }
    }
    if (!bc1.empty())
    {
      for (Eigen::Index j = 0; j < Ae.cols(); ++j)
      {
        if (bc1[dmap1[j]])
          Ae.col(j).setZero();
      }
    }

    ierr = MatSetValuesLocal(A, dmap0.size(), dmap0.data(), dmap1.size(),
                             dmap1.data(), Ae.data(), ADD_VALUES);
#ifdef DEBUG
    if (ierr != 0)
      la::petsc_error(ierr, __FILE__, "MatSetValuesLocal");
#endif
  }
}
//-----------------------------------------------------------------------------
