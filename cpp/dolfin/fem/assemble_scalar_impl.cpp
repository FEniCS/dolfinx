// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_scalar_impl.h"
#include "Form.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/types.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <petscsys.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
PetscScalar dolfin::fem::impl::assemble_scalar(const dolfin::fem::Form& M)
{
  assert(M.mesh());
  const mesh::Mesh& mesh = *M.mesh();

  // Prepare coefficients
  const FormCoefficients& coefficients = M.coeffs();
  std::vector<const function::Function*> coeff_fn(coefficients.size());
  for (int i = 0; i < coefficients.size(); ++i)
    coeff_fn[i] = coefficients.get(i).get();
  std::vector<int> c_offsets = coefficients.offsets();

  const FormIntegrals& integrals = M.integrals();
  using type = fem::FormIntegrals::Type;
  PetscScalar value = 0.0;
  for (int i = 0; i < integrals.num_integrals(type::cell); ++i)
  {
    auto& fn = integrals.get_tabulate_tensor_fn_cell(i);
    const std::vector<std::int32_t>& active_cells
        = integrals.integral_domains(type::cell, i);
    value += fem::impl::assemble_cells(mesh, active_cells, fn, coeff_fn,
                                       c_offsets);
  }

  for (int i = 0; i < integrals.num_integrals(type::exterior_facet); ++i)
  {
    auto& fn = integrals.get_tabulate_tensor_fn_exterior_facet(i);
    const std::vector<std::int32_t>& active_facets = integrals.integral_domains(
        fem::FormIntegrals::Type::exterior_facet, i);
    value += fem::impl::assemble_exterior_facets(mesh, active_facets, fn,
                                                 coeff_fn, c_offsets);
  }

  for (int i = 0; i < integrals.num_integrals(type::interior_facet); ++i)
  {
    auto& fn = integrals.get_tabulate_tensor_fn_interior_facet(i);
    const std::vector<std::int32_t>& active_facets = integrals.integral_domains(
        fem::FormIntegrals::Type::interior_facet, i);
    value += fem::impl::assemble_interior_facets(mesh, active_facets, fn,
                                                 coeff_fn, c_offsets);
  }

  return value;
}
//-----------------------------------------------------------------------------
PetscScalar fem::impl::assemble_cells(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_cells,
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             const int*, const int*)>& fn,
    std::vector<const function::Function*> coefficients,
    const std::vector<int>& offsets)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();
  mesh.create_entities(tdim);

  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = mesh.coordinate_dofs().entity_points();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> pos_g
      = connectivity_g.entity_positions();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().points();

  // Create data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(offsets.back());

  // Iterate over all cells
  PetscScalar cell_value, value(0);

  for (const auto& cell_index : active_cells)
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

    fn(&cell_value, coeff_array.data(), coordinate_dofs.data(), NULL, NULL);
    value += cell_value;
  }

  return value;
}
//-----------------------------------------------------------------------------
PetscScalar fem::impl::assemble_exterior_facets(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             const int*, const int*)>& fn,
    std::vector<const function::Function*> coefficients,
    const std::vector<int>& offsets)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();
  mesh.create_entities(tdim - 1);
  mesh.create_connectivity(tdim - 1, tdim);

  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = mesh.coordinate_dofs().entity_points();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> pos_g
      = connectivity_g.entity_positions();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().points();

  // Creat data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(offsets.back());

  // Iterate over all facets
  PetscScalar cell_value, value(0);
  for (const auto& facet_index : active_facets)
  {
    const mesh::Facet facet(mesh, facet_index);

    assert(facet.num_global_entities(tdim) == 1);

    // TODO: check ghosting sanity?

    // Create attached cell
    const mesh::Cell cell(mesh, facet.entities(tdim)[0]);

    // Get local index of facet with respect to the cell
    const int local_facet = cell.index(facet);
    const int orient = 1;

    // Get cell vertex coordinates
    const int cell_index = cell.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Update coefficients
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
      coefficients[i]->restrict(coeff_array.data() + offsets[i], cell,
                                coordinate_dofs);
    }

    fn(&cell_value, coeff_array.data(), coordinate_dofs.data(), &local_facet,
       &orient);
    value += cell_value;
  }

  return value;
}
//-----------------------------------------------------------------------------
PetscScalar fem::impl::assemble_interior_facets(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             const int*, const int*)>& fn,
    std::vector<const function::Function*> coefficients,
    const std::vector<int>& offsets)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();
  mesh.create_entities(tdim - 1);
  mesh.create_connectivity(tdim - 1, tdim);

  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = mesh.coordinate_dofs().entity_points();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> pos_g
      = connectivity_g.entity_positions();
  const Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().points();

  // Creat data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(2 * num_dofs_g, gdim);
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(2 * offsets.back());

  // Iterate over all facets
  PetscScalar cell_value, value(0);
  for (const auto& facet_index : active_facets)
  {
    const mesh::Facet facet(mesh, facet_index);

    assert(facet.num_global_entities(tdim) == 2);

    // TODO: check ghosting sanity?

    // Create attached cell
    const mesh::Cell cell0(mesh, facet.entities(tdim)[0]);
    const mesh::Cell cell1(mesh, facet.entities(tdim)[1]);

    // Get local index of facet with respect to the cell
    const int local_facet[2] = {cell0.index(facet), cell1.index(facet)};
    const int orient[2] = {1, 1};

    // Get cell vertex coordinates
    const int cell0_index = cell0.index();
    const int cell1_index = cell1.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
      {
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell0_index] + i], j);
        coordinate_dofs(i + num_dofs_g, j)
            = x_g(cell_g[pos_g[cell1_index] + i], j);
      }

    // Update coefficients
    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        coordinate_dofs0(coordinate_dofs.data(), num_dofs_g, gdim);

    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        coordinate_dofs1(coordinate_dofs.data() + num_dofs_g * gdim, num_dofs_g,
                         gdim);
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
      coefficients[i]->restrict(coeff_array.data() + offsets[i], cell0,
                                coordinate_dofs0);
      coefficients[i]->restrict(coeff_array.data() + offsets.back()
                                    + offsets[i],
                                cell1, coordinate_dofs1);
    }

    fn(&cell_value, coeff_array.data(), coordinate_dofs.data(), local_facet,
       orient);
    value += cell_value;
  }

  return value;
}
//-----------------------------------------------------------------------------
