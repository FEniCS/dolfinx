// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_scalar_impl.h"
#include "DofMap.h"
#include "Form.h"
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/types.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/CoordinateDofs.h>
#include <dolfin/mesh/Geometry.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEntity.h>
#include <dolfin/mesh/MeshIterator.h>
#include <petscsys.h>

using namespace dolfin;
using namespace dolfin::fem;

namespace
{
//-----------------------------------------------------------------------------
Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
pack_coefficients(const fem::Form& L)
{
  // Get form coefficient offsets amd dofmaps
  const FormCoefficients& coefficients = L.coefficients();
  const std::vector<int> offsets = coefficients.offsets();
  std::vector<const fem::DofMap*> dofmaps(coefficients.size());
  for (int i = 0; i < coefficients.size(); ++i)
    dofmaps[i] = coefficients.get(i)->function_space()->dofmap().get();

  // Get mesh
  assert(L.mesh());
  const mesh::Mesh mesh = *L.mesh();
  const int tdim = mesh.topology().dim();

  // Unwrap PETSc vectors
  std::vector<const PetscScalar*> v(coefficients.size(), nullptr);
  std::vector<Vec> x(coefficients.size(), nullptr),
      x_local(coefficients.size(), nullptr);
  for (std::size_t i = 0; i < v.size(); ++i)
  {
    x[i] = coefficients.get(i)->vector().vec();
    VecGhostGetLocalForm(x[i], &x_local[i]);
    VecGetArrayRead(x_local[i], &v[i]);
  }

  // Copy data into coefficient array
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> c(
      mesh.num_entities(tdim), offsets.back());
  for (int cell = 0; cell < mesh.num_entities(tdim); ++cell)
  {
    auto c_cell = c.row(cell);
    for (std::size_t coeff = 0; coeff < dofmaps.size(); ++coeff)
    {
      auto dofs = dofmaps[coeff]->cell_dofs(cell);
      const PetscScalar* _v = v[coeff];
      for (Eigen::Index k = 0; k < dofs.size(); ++k)
        c_cell(k + offsets[coeff]) = _v[dofs[k]];
    }
  }

  // Restore PETSc vectors
  for (std::size_t i = 0; i < v.size(); ++i)
  {
    VecRestoreArrayRead(x_local[i], &v[i]);
    VecGhostRestoreLocalForm(x[i], &x_local[i]);
  }

  return c;
}
//-----------------------------------------------------------------------------

} // namespace

//-----------------------------------------------------------------------------
PetscScalar dolfin::fem::impl::assemble_scalar(const dolfin::fem::Form& M)
{
  assert(M.mesh());
  const mesh::Mesh& mesh = *M.mesh();

  // Prepare constants
  const std::vector<
      std::pair<std::string, std::shared_ptr<const function::Constant>>>
      constants = M.constants();

  std::vector<PetscScalar> constant_values;
  for (auto const& constant : constants)
  {
    // Get underlying data array of this Constant
    const std::vector<PetscScalar>& array = constant.second->value;

    constant_values.insert(constant_values.end(), array.data(),
                           array.data() + array.size());
  }

  // Prepare coefficients
  const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::RowMajor>
      coeffs = pack_coefficients(M);

  const FormIntegrals& integrals = M.integrals();
  using type = fem::FormIntegrals::Type;
  PetscScalar value = 0.0;
  for (int i = 0; i < integrals.num_integrals(type::cell); ++i)
  {
    auto& fn = integrals.get_tabulate_tensor_function(type::cell, i);
    const std::vector<std::int32_t>& active_cells
        = integrals.integral_domains(type::cell, i);
    value += fem::impl::assemble_cells(mesh, active_cells, fn, coeffs,
                                       constant_values);
  }

  for (int i = 0; i < integrals.num_integrals(type::exterior_facet); ++i)
  {
    auto& fn = integrals.get_tabulate_tensor_function(type::exterior_facet, i);
    const std::vector<std::int32_t>& active_facets
        = integrals.integral_domains(type::exterior_facet, i);
    value += fem::impl::assemble_exterior_facets(mesh, active_facets, fn,
                                                 coeffs, constant_values);
  }

  for (int i = 0; i < integrals.num_integrals(type::interior_facet); ++i)
  {
    const std::vector<int> c_offsets = M.coefficients().offsets();
    auto& fn = integrals.get_tabulate_tensor_function(type::interior_facet, i);
    const std::vector<std::int32_t>& active_facets
        = integrals.integral_domains(type::interior_facet, i);
    value += fem::impl::assemble_interior_facets(
        mesh, active_facets, fn, coeffs, c_offsets, constant_values);
  }

  return value;
}
//-----------------------------------------------------------------------------
PetscScalar fem::impl::assemble_cells(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_cells,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const int*)>& fn,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const std::vector<PetscScalar> constant_values)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();
  mesh.create_entities(tdim);

  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = mesh.coordinate_dofs().entity_points();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& pos_g
      = connectivity_g.entity_positions();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().points();

  // Create data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);

  // Iterate over all cells
  const int orientation = 0;
  PetscScalar value(0);
  for (const auto& cell_index : active_cells)
  {
    const mesh::MeshEntity cell(mesh, tdim, cell_index);

    // Get cell coordinates/geometry
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    auto coeff_cell = coeffs.row(cell_index);
    fn(&value, coeff_cell.data(), constant_values.data(),
       coordinate_dofs.data(), nullptr, &orientation);
  }

  return value;
}
//-----------------------------------------------------------------------------
PetscScalar fem::impl::assemble_exterior_facets(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const int*)>& fn,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const std::vector<PetscScalar> constant_values)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();
  mesh.create_entities(tdim - 1);
  mesh.create_connectivity(tdim - 1, tdim);

  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = mesh.coordinate_dofs().entity_points();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& pos_g
      = connectivity_g.entity_positions();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().points();

  // Creat data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);

  // Iterate over all facets
  PetscScalar value(0);
  for (const auto& facet_index : active_facets)
  {
    const mesh::MeshEntity facet(mesh, tdim - 1, facet_index);

    // TODO: check ghosting sanity?

    // Create attached cell
    const mesh::MeshEntity cell(mesh, tdim, facet.entities(tdim)[0]);

    // Get local index of facet with respect to the cell
    const int local_facet = cell.index(facet);
    const int orient = 0;

    // Get cell vertex coordinates
    const int cell_index = cell.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    auto coeff_cell = coeffs.row(cell_index);
    fn(&value, coeff_cell.data(), constant_values.data(),
       coordinate_dofs.data(), &local_facet, &orient);
  }

  return value;
}
//-----------------------------------------------------------------------------
PetscScalar fem::impl::assemble_interior_facets(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const int*)>& fn,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const std::vector<int>& offsets,
    const std::vector<PetscScalar> constant_values)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();
  mesh.create_entities(tdim - 1);
  mesh.create_connectivity(tdim - 1, tdim);

  // Prepare cell geometry
  const mesh::Connectivity& connectivity_g
      = mesh.coordinate_dofs().entity_points();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& pos_g
      = connectivity_g.entity_positions();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& cell_g
      = connectivity_g.connections();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.size(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().points();

  // Creat data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(2 * num_dofs_g, gdim);
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(2 * offsets.back());
  assert(offsets.back() == coeffs.cols());

  // Iterate over all facets
  PetscScalar value(0);
  for (const auto& facet_index : active_facets)
  {
    const mesh::MeshEntity facet(mesh, tdim - 1, facet_index);

    // TODO: check ghosting sanity?

    // Create attached cell
    const mesh::MeshEntity cell0(mesh, tdim, facet.entities(tdim)[0]);
    const mesh::MeshEntity cell1(mesh, tdim, facet.entities(tdim)[1]);

    // Get local index of facet with respect to the cell
    const int local_facet[2] = {cell0.index(facet), cell1.index(facet)};
    const int orient[2] = {0, 0};

    // Get cell vertex coordinates
    const int cell_index0 = cell0.index();
    const int cell_index1 = cell1.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
      {
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index0] + i], j);
        coordinate_dofs(i + num_dofs_g, j)
            = x_g(cell_g[pos_g[cell_index1] + i], j);
      }

    // Update coefficients
    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        coordinate_dofs0(coordinate_dofs.data(), num_dofs_g, gdim);

    Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::RowMajor>>
        coordinate_dofs1(coordinate_dofs.data() + num_dofs_g * gdim, num_dofs_g,
                         gdim);

    // Layout for the restricted coefficients is flattened
    // w[coefficient][restriction][dof]
    auto coeff_cell0 = coeffs.row(cell_index0);
    auto coeff_cell1 = coeffs.row(cell_index1);

    // Loop over coefficients
    for (std::size_t i = 0; i < offsets.size() - 1; ++i)
    {
      // Loop over entries for coefficient i
      for (int j = 0; j < offsets[i + 1]; ++j)
      {
        coeff_array(2 * offsets[i] + j) = coeff_cell0(offsets[i] + j);
        coeff_array(offsets[i + 1] + offsets[i] + j)
            = coeff_cell1(offsets[i] + j);
      }
    }

    fn(&value, coeff_array.data(), constant_values.data(),
       coordinate_dofs.data(), local_facet, orient);
  }

  return value;
}
//-----------------------------------------------------------------------------
