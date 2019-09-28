// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_vector_impl.h"
#include "DirichletBC.h"
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
void _restrict(const fem::DofMap& dofmap, const Vec x, int cell_index,
               PetscScalar* w)
{
  assert(w);
  auto dofs = dofmap.cell_dofs(cell_index);

  // Pick values from vector(s)
  la::VecReadWrapper v(x);
  Eigen::Map<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> _v = v.x;
  for (Eigen::Index i = 0; i < dofs.size(); ++i)
    w[i] = _v[dofs[i]];
}
//-----------------------------------------------------------------------------
// Implementation of bc application
void _lift_bc_cells(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& a,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
        bc_values1,
    const std::vector<bool>& bc_markers1,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x0,
    double scale)
{
  assert(a.rank() == 2);

  // Get mesh from form
  assert(a.mesh());
  const mesh::Mesh& mesh = *a.mesh();

  // Get dofmap for columns and rows of a
  assert(a.function_space(0));
  assert(a.function_space(0)->dofmap());
  assert(a.function_space(1));
  assert(a.function_space(1)->dofmap());
  const fem::DofMap& dofmap0 = *a.function_space(0)->dofmap();
  const fem::DofMap& dofmap1 = *a.function_space(1)->dofmap();

  // TODO: simplify and move elsewhere
  // Manage coefficients
  const FormCoefficients& coefficients = a.coefficients();
  std::vector<std::uint32_t> n = {0};
  std::vector<const function::Function*> coefficients_ptr(coefficients.size());
  for (int i = 0; i < coefficients.size(); ++i)
  {
    coefficients_ptr[i] = coefficients.get(i).get();
    n.push_back(
        n.back()
        + coefficients_ptr[i]->function_space()->element()->space_dimension());
  }
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(n.back());

  const std::function<void(PetscScalar*, const PetscScalar*, const PetscScalar*,
                           const double*, const int*, const int*)>& fn
      = a.integrals().get_tabulate_tensor_function(FormIntegrals::Type::cell,
                                                   0);

  // Prepare cell geometry
  const int gdim = mesh.geometry().dim();
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

  // Data structures used in bc application
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;

  const std::vector<
      std::pair<std::string, std::shared_ptr<const function::Constant>>>
      constants = a.constants();

  std::vector<PetscScalar> constant_values;
  for (auto const& constant : constants)
  {
    // Get underlying data array of this Constant
    const std::vector<PetscScalar>& array = constant.second->value;

    constant_values.insert(constant_values.end(), array.data(),
                           array.data() + array.size());
  }

  // Iterate over all cells
  const int tdim = mesh.geometry().dim();
  const int orient = 0;
  for (const mesh::MeshEntity& cell : mesh::MeshRange(mesh, tdim))
  {
    // Get dof maps for cell
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap1
        = dofmap1.cell_dofs(cell.index());

    // Check if bc is applied to cell
    bool has_bc = false;
    for (Eigen::Index j = 0; j < dmap1.size(); ++j)
    {
      if (bc_markers1[dmap1[j]])
      {
        has_bc = true;
        break;
      }
    }

    if (!has_bc)
      continue;

    // Get cell vertex coordinates
    const int cell_index = cell.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Size data structure for assembly
    const Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap0
        = dofmap0.cell_dofs(cell.index());

    // TODO: Move gathering of coefficients outside of main assembly
    // loop
    // Update coefficients
    for (int i = 0; i < coefficients.size(); ++i)
    {
      _restrict(*coefficients_ptr[i]->function_space()->dofmap(),
                coefficients_ptr[i]->vector().vec(), cell.index(),
                coeff_array.data() + n[i]);
    }

    Ae.setZero(dmap0.size(), dmap1.size());
    fn(Ae.data(), coeff_array.data(), constant_values.data(),
       coordinate_dofs.data(), nullptr, &orient);

    // Size data structure for assembly
    be.setZero(dmap0.size());
    for (Eigen::Index j = 0; j < dmap1.size(); ++j)
    {
      const PetscInt jj = dmap1[j];
      if (bc_markers1[jj])
      {
        const PetscScalar bc = bc_values1[jj];
        if (x0.rows() > 0)
          be -= Ae.col(j) * scale * (bc - x0[jj]);
        else
          be -= Ae.col(j) * scale * bc;
      }
    }

    for (Eigen::Index k = 0; k < dmap0.size(); ++k)
      b[dmap0[k]] += be[k];
  }
}
//----------------------------------------------------------------------------
void _lift_bc_exterior_facets(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& a,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
        bc_values1,
    const std::vector<bool>& bc_markers1,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x0,
    double scale)
{
  assert(a.rank() == 2);

  // Get mesh from form
  assert(a.mesh());
  const mesh::Mesh& mesh = *a.mesh();

  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();
  mesh.create_entities(tdim - 1);
  mesh.create_connectivity(tdim - 1, tdim);

  // Get dofmap for columns and rows of a
  assert(a.function_space(0));
  assert(a.function_space(0)->dofmap());
  assert(a.function_space(1));
  assert(a.function_space(1)->dofmap());
  const fem::DofMap& dofmap0 = *a.function_space(0)->dofmap();
  const fem::DofMap& dofmap1 = *a.function_space(1)->dofmap();

  // TODO: simplify and move elsewhere
  // Manage coefficients
  const FormCoefficients& coefficients = a.coefficients();
  std::vector<std::uint32_t> n = {0};
  std::vector<const function::Function*> coefficients_ptr(coefficients.size());
  for (int i = 0; i < coefficients.size(); ++i)
  {
    coefficients_ptr[i] = coefficients.get(i).get();
    n.push_back(
        n.back()
        + coefficients_ptr[i]->function_space()->element()->space_dimension());
  }
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(n.back());

  const std::function<void(PetscScalar*, const PetscScalar*, const PetscScalar*,
                           const double*, const int*, const int*)>& fn
      = a.integrals().get_tabulate_tensor_function(
          FormIntegrals::Type::exterior_facet, 0);

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

  // Data structures used in bc application
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;

  const std::vector<
      std::pair<std::string, std::shared_ptr<const function::Constant>>>
      constants = a.constants();

  std::vector<PetscScalar> constant_values;
  for (auto const& constant : constants)
  {
    // Get underlying data array of this Constant
    const std::vector<PetscScalar>& array = constant.second->value;

    constant_values.insert(constant_values.end(), array.data(),
                           array.data() + array.size());
  }

  // Iterate over all cells
  std::shared_ptr<const mesh::Connectivity> connectivity
      = mesh.topology().connectivity(tdim - 1, tdim);
  assert(connectivity);
  for (const mesh::MeshEntity& facet : mesh::MeshRange(mesh, tdim - 1))
  {
    // Move to next facet if this one is an interior facet
    if (connectivity->size_global(facet.index()) != 1)
      continue;

    // FIXME: sort out ghosts

    // Create attached cell
    mesh::MeshEntity cell(mesh, tdim, facet.entities(tdim)[0]);

    // Get local index of facet with respect to the cell
    const int local_facet = cell.index(facet);
    const int orient = 0;

    // Get dof maps for cell
    const Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap1
        = dofmap1.cell_dofs(cell.index());

    // Check if bc is applied to cell
    bool has_bc = false;
    for (Eigen::Index j = 0; j < dmap1.size(); ++j)
    {
      if (bc_markers1[dmap1[j]])
      {
        has_bc = true;
        break;
      }
    }

    if (!has_bc)
      continue;

    // Get cell vertex coordinates
    const int cell_index = cell.index();
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Size data structure for assembly
    const Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap0
        = dofmap0.cell_dofs(cell.index());

    // TODO: Move gathering of coefficients outside of main assembly
    // loop
    // Update coefficients
    for (int i = 0; i < coefficients.size(); ++i)
    {
      _restrict(*coefficients_ptr[i]->function_space()->dofmap(),
                coefficients_ptr[i]->vector().vec(), cell.index(),
                coeff_array.data() + n[i]);
    }

    Ae.setZero(dmap0.size(), dmap1.size());
    fn(Ae.data(), coeff_array.data(), constant_values.data(),
       coordinate_dofs.data(), &local_facet, &orient);

    // Size data structure for assembly
    be.setZero(dmap0.size());
    for (Eigen::Index j = 0; j < dmap1.size(); ++j)
    {
      const PetscInt jj = dmap1[j];
      if (bc_markers1[jj])
      {
        const PetscScalar bc = bc_values1[jj];
        if (x0.rows() > 0)
          be -= Ae.col(j) * scale * (bc - x0[jj]);
        else
          be -= Ae.col(j) * scale * bc;
      }
    }

    for (Eigen::Index k = 0; k < dmap0.size(); ++k)
      b[dmap0[k]] += be[k];
  }
}
} // namespace

//-----------------------------------------------------------------------------
void fem::impl::assemble_vector(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& L)
{
  assert(L.mesh());
  const mesh::Mesh& mesh = *L.mesh();

  // Get dofmap data
  const fem::DofMap& dofmap = *L.function_space(0)->dofmap();
  Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dof_array
      = dofmap.dof_array();

  assert(dofmap.element_dof_layout);
  const int num_dofs_per_cell = dofmap.element_dof_layout->num_dofs();

  // Prepare coefficients
  const FormCoefficients& coefficients = L.coefficients();
  std::vector<const function::Function*> coeff_fn(coefficients.size());
  for (int i = 0; i < coefficients.size(); ++i)
    coeff_fn[i] = coefficients.get(i).get();
  std::vector<int> c_offsets = coefficients.offsets();

  const std::vector<
      std::pair<std::string, std::shared_ptr<const function::Constant>>>
      constants = L.constants();

  std::vector<PetscScalar> constant_values;
  for (auto const& constant : constants)
  {
    // Get underlying data array of this Constant
    const std::vector<PetscScalar>& array = constant.second->value;

    constant_values.insert(constant_values.end(), array.data(),
                           array.data() + array.size());
  }

  const FormIntegrals& integrals = L.integrals();
  using type = fem::FormIntegrals::Type;
  for (int i = 0; i < integrals.num_integrals(type::cell); ++i)
  {
    auto& fn
        = integrals.get_tabulate_tensor_function(FormIntegrals::Type::cell, i);
    const std::vector<std::int32_t>& active_cells
        = integrals.integral_domains(type::cell, i);
    fem::impl::assemble_cells(b, mesh, active_cells, dof_array,
                              num_dofs_per_cell, fn, coeff_fn, c_offsets,
                              constant_values);
  }

  for (int i = 0; i < integrals.num_integrals(type::exterior_facet); ++i)
  {
    const auto& fn = integrals.get_tabulate_tensor_function(
        FormIntegrals::Type::exterior_facet, i);
    const std::vector<std::int32_t>& active_facets
        = integrals.integral_domains(type::exterior_facet, i);
    fem::impl::assemble_exterior_facets(b, mesh, active_facets, dofmap, fn,
                                        coeff_fn, c_offsets, constant_values);
  }

  for (int i = 0; i < integrals.num_integrals(type::interior_facet); ++i)
  {
    const auto& fn = integrals.get_tabulate_tensor_function(
        FormIntegrals::Type::interior_facet, i);
    const std::vector<std::int32_t>& active_facets
        = integrals.integral_domains(type::interior_facet, i);
    fem::impl::assemble_interior_facets(b, mesh, active_facets, dofmap, fn,
                                        coeff_fn, c_offsets, constant_values);
  }
}
//-----------------------------------------------------------------------------
void fem::impl::assemble_cells(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_cells,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dofmap,
    int num_dofs_per_cell,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const int*)>& kernel,
    const std::vector<const function::Function*>& coefficients,
    const std::vector<int>& offsets,
    const std::vector<PetscScalar> constant_values)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();

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
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be(num_dofs_per_cell);
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(offsets.back());

  // Iterate over active cells
  const int orientation = 0;
  for (std::int32_t cell_index : active_cells)
  {
    const mesh::MeshEntity cell(mesh, tdim, cell_index);

    // Get cell coordinates/geometry
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // FIXME: Move this outside of inner assembly loop
    // Update coefficients
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
      _restrict(*coefficients[i]->function_space()->dofmap(),
                coefficients[i]->vector().vec(), cell.index(),
                coeff_array.data() + offsets[i]);
    }

    // Tabulate vector for cell
    be.setZero();
    kernel(be.data(), coeff_array.data(), constant_values.data(),
           coordinate_dofs.data(), nullptr, &orientation);

    // Add local cell vector to global vector
    for (Eigen::Index i = 0; i < num_dofs_per_cell; ++i)
      b[dofmap[cell_index * num_dofs_per_cell + i]] += be[i];
  }
}
//-----------------------------------------------------------------------------
void fem::impl::assemble_exterior_facets(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const fem::DofMap& dofmap,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const int*)>& fn,
    const std::vector<const function::Function*>& coefficients,
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
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(offsets.back());

  for (const auto& facet_index : active_facets)
  {
    const mesh::MeshEntity facet(mesh, tdim - 1, facet_index);

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

    // Get dof map for cell
    const Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap
        = dofmap.cell_dofs(cell.index());

    // TODO: Move gathering of coefficients outside of main assembly
    // loop
    // Update coefficients
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
      coefficients[i]->restrict(cell, coordinate_dofs,
                                coeff_array.data() + offsets[i]);
    }

    // Tabulate element vector
    be.setZero(dmap.size());
    fn(be.data(), coeff_array.data(), constant_values.data(),
       coordinate_dofs.data(), &local_facet, &orient);

    // Add element vector to global vector
    for (Eigen::Index i = 0; i < dmap.size(); ++i)
      b[dmap[i]] += be[i];
  }
}
//-----------------------------------------------------------------------------
void fem::impl::assemble_interior_facets(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const fem::DofMap& dofmap,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const int*)>& fn,
    const std::vector<const function::Function*>& coefficients,
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
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(2 * offsets.back());

  for (const auto& facet_index : active_facets)
  {
    const mesh::MeshEntity facet(mesh, tdim - 1, facet_index);

    // assert(facet.num_global_entities(tdim) == 2);

    // TODO: check ghosting sanity?

    // Create attached cells
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

    // Get dofmaps for cell
    const Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap0
        = dofmap.cell_dofs(cell_index0);
    const Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap1
        = dofmap.cell_dofs(cell_index1);

    // TODO: Move gathering of coefficients outside of main assembly
    // loop
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
      coefficients[i]->restrict(cell0, coordinate_dofs0,
                                coeff_array.data() + offsets[i]);
      coefficients[i]->restrict(cell1, coordinate_dofs1,
                                coeff_array.data() + offsets.back()
                                    + offsets[i]);
    }

    // Tabulate element vector
    be.setZero(dmap0.size() + dmap1.size());
    fn(be.data(), coeff_array.data(), constant_values.data(),
       coordinate_dofs.data(), local_facet, orient);

    // Add element vector to global vector
    for (Eigen::Index i = 0; i < dmap0.size(); ++i)
      b[dmap0[i]] += be[i];
    for (Eigen::Index i = 0; i < dmap1.size(); ++i)
      b[dmap1[i]] += be[i + dmap0.size()];
  }
}
//-----------------------------------------------------------------------------
void fem::impl::apply_lifting(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const std::vector<std::shared_ptr<const Form>> a,
    std::vector<std::vector<std::shared_ptr<const DirichletBC>>> bcs1,
    std::vector<Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>>
        x0,
    double scale)
{
  // FIXME: make changes to reactivate this check
  // if (!x0.empty() and x0.size() != a.size())
  //   throw std::runtime_error("Mismatch in size between x0 and a in
  //   assembler.");
  if (a.size() != bcs1.size())
  {
    throw std::runtime_error(
        "Mismatch in size between a and bcs in assembler.");
  }

  for (std::size_t j = 0; j < a.size(); ++j)
  {
    std::vector<bool> bc_markers1;
    Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> bc_values1;
    if (a[j] and !bcs1[j].empty())
    {
      auto V1 = a[j]->function_space(1);
      assert(V1);
      auto map1 = V1->dofmap()->index_map;
      assert(map1);
      const int crange
          = map1->block_size * (map1->size_local() + map1->num_ghosts());
      bc_markers1.assign(crange, false);
      bc_values1 = Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>::Zero(crange);
      for (std::shared_ptr<const DirichletBC>& bc : bcs1[j])
      {
        bc->mark_dofs(bc_markers1);
        bc->dof_values(bc_values1);
      }

      // Modify (apply lifting) vector
      if (!x0.empty())
        fem::impl::lift_bc(b, *a[j], bc_values1, bc_markers1, x0[j], scale);
      else
        fem::impl::lift_bc(b, *a[j], bc_values1, bc_markers1, scale);
    }
  }
}
//-----------------------------------------------------------------------------
void fem::impl::lift_bc(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& a,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
        bc_values1,
    const std::vector<bool>& bc_markers1, double scale)
{
  // FIXME: add lifting over exterior facets

  const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> x0(0);
  if (a.integrals().num_integrals(fem::FormIntegrals::Type::cell) > 0)
    _lift_bc_cells(b, a, bc_values1, bc_markers1, x0, scale);
  if (a.integrals().num_integrals(fem::FormIntegrals::Type::exterior_facet) > 0)
    _lift_bc_exterior_facets(b, a, bc_values1, bc_markers1, x0, scale);
}
//-----------------------------------------------------------------------------
void fem::impl::lift_bc(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& a,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>
        bc_values1,
    const std::vector<bool>& bc_markers1,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x0,
    double scale)
{
  if (b.size() != x0.size())
  {
    throw std::runtime_error(
        "Vector size mismatch in modification for boundary conditions.");
  }

  if (a.integrals().num_integrals(fem::FormIntegrals::Type::cell) > 0)
    _lift_bc_cells(b, a, bc_values1, bc_markers1, x0, scale);
  if (a.integrals().num_integrals(fem::FormIntegrals::Type::exterior_facet) > 0)
    _lift_bc_exterior_facets(b, a, bc_values1, bc_markers1, x0, scale);
}
//-----------------------------------------------------------------------------
