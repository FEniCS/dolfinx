// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_vector_impl.h"
#include "DirichletBC.h"
#include "DofMap.h"
#include "Form.h"
#include "utils.h"
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
// Implementation of bc application
void _lift_bc_cells(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& a,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>&
        bc_values1,
    const std::vector<bool>& bc_markers1,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>& x0,
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

  // Prepare coefficients
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coeffs = pack_coefficients(a);

  const std::function<void(PetscScalar*, const PetscScalar*, const PetscScalar*,
                           const double*, const int*, const int*)>& fn
      = a.integrals().get_tabulate_tensor_function(FormIntegrals::Type::cell,
                                                   0);

  // Prepare cell geometry
  const int gdim = mesh.geometry().dim();
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
    auto dmap1 = dofmap1.cell_dofs(cell.index());

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

    const int cell_index = cell.index();

    // Get cell vertex coordinates
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Size data structure for assembly
    auto dmap0 = dofmap0.cell_dofs(cell_index);

    auto coeff_array = coeffs.row(cell_index);
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
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>&
        bc_values1,
    const std::vector<bool>& bc_markers1,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>& x0,
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

  // Prepare coefficients
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coeffs = pack_coefficients(a);

  const std::function<void(PetscScalar*, const PetscScalar*, const PetscScalar*,
                           const double*, const int*, const int*)>& fn
      = a.integrals().get_tabulate_tensor_function(
          FormIntegrals::Type::exterior_facet, 0);

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
    const std::int32_t cell_index = facet.entities(tdim)[0];

    // Get local index of facet with respect to the cell
    mesh::MeshEntity cell(mesh, tdim, cell_index);
    const int local_facet = cell.index(facet);
    const int orient = 0;

    // Get dof maps for cell
    auto dmap1 = dofmap1.cell_dofs(cell_index);

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
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Size data structure for assembly
    auto dmap0 = dofmap0.cell_dofs(cell_index);

    // TODO: Move gathering of coefficients outside of main assembly
    // loop

    auto coeff_array = coeffs.row(cell_index);
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
  const Eigen::Array<PetscInt, Eigen::Dynamic, 1>& dof_array
      = dofmap.dof_array();

  assert(dofmap.element_dof_layout);
  const int num_dofs_per_cell = dofmap.element_dof_layout->num_dofs();

  // Prepare constants
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

  // Prepare coefficients
  const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::RowMajor>
      coeffs = pack_coefficients(L);

  const FormIntegrals& integrals = L.integrals();
  using type = fem::FormIntegrals::Type;
  for (int i = 0; i < integrals.num_integrals(type::cell); ++i)
  {
    auto& fn
        = integrals.get_tabulate_tensor_function(FormIntegrals::Type::cell, i);
    const std::vector<std::int32_t>& active_cells
        = integrals.integral_domains(type::cell, i);
    fem::impl::assemble_cells(b, mesh, active_cells, dof_array,
                              num_dofs_per_cell, fn, coeffs, constant_values);
  }

  for (int i = 0; i < integrals.num_integrals(type::exterior_facet); ++i)
  {
    const auto& fn = integrals.get_tabulate_tensor_function(
        FormIntegrals::Type::exterior_facet, i);
    const std::vector<std::int32_t>& active_facets
        = integrals.integral_domains(type::exterior_facet, i);
    fem::impl::assemble_exterior_facets(b, mesh, active_facets, dofmap, fn,
                                        coeffs, constant_values);
  }

  for (int i = 0; i < integrals.num_integrals(type::interior_facet); ++i)
  {
    const std::vector<int> c_offsets = L.coefficients().offsets();
    const auto& fn = integrals.get_tabulate_tensor_function(
        FormIntegrals::Type::interior_facet, i);
    const std::vector<std::int32_t>& active_facets
        = integrals.integral_domains(type::interior_facet, i);
    fem::impl::assemble_interior_facets(b, mesh, active_facets, dofmap, fn,
                                        coeffs, c_offsets, constant_values);
  }
}
//-----------------------------------------------------------------------------
void fem::impl::assemble_cells(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_cells,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofmap,
    int num_dofs_per_cell,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const int*)>& kernel,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const std::vector<PetscScalar> constant_values)
{
  const int gdim = mesh.geometry().dim();

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

  // Iterate over active cells
  const int orientation = 0;
  for (std::int32_t cell_index : active_cells)
  {
    // Get cell coordinates/geometry
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Tabulate vector for cell
    auto coeff_cell = coeffs.row(cell_index);
    be.setZero();
    kernel(be.data(), coeff_cell.data(), constant_values.data(),
           coordinate_dofs.data(), nullptr, &orientation);

    // Scatter cell vector to 'global' vector array
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
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;
  for (const auto& facet_index : active_facets)
  {
    const mesh::MeshEntity facet(mesh, tdim - 1, facet_index);

    // Get index of first attached cell
    const std::int32_t cell_index = facet.entities(tdim)[0];

    // FIXME: See if creation of MeshEntity can be removed
    // Get local index of facet with respect to the cell
    const mesh::MeshEntity cell(mesh, tdim, cell_index);
    const int local_facet = cell.index(facet);
    const int orient = 0;

    // Get cell vertex coordinates
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    // Get dof map for cell
    auto dmap = dofmap.cell_dofs(cell_index);

    // Tabulate element vector
    auto coeff_cell = coeffs.row(cell_index);
    be.setZero(dmap.size());
    fn(be.data(), coeff_cell.data(), constant_values.data(),
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
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(2 * offsets.back());
  assert(offsets.back() == coeffs.cols());

  for (const auto& facet_index : active_facets)
  {
    const mesh::MeshEntity facet(mesh, tdim - 1, facet_index);

    // assert(facet.num_global_entities(tdim) == 2);
    // TODO: check ghosting sanity?

    // Get attached cell indices
    const std::int32_t cell_index0 = facet.entities(tdim)[0];
    const std::int32_t cell_index1 = facet.entities(tdim)[1];

    // Create attached cells
    const mesh::MeshEntity cell0(mesh, tdim, facet.entities(tdim)[0]);
    const mesh::MeshEntity cell1(mesh, tdim, facet.entities(tdim)[1]);
    const int local_facet[2] = {cell0.index(facet), cell1.index(facet)};

    // Orientation
    const int orient[2] = {0, 0};

    // Get cell vertex coordinates
    for (int i = 0; i < num_dofs_g; ++i)
    {
      for (int j = 0; j < gdim; ++j)
      {
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index0] + i], j);
        coordinate_dofs(i + num_dofs_g, j)
            = x_g(cell_g[pos_g[cell_index1] + i], j);
      }
    }

    // Get dofmaps for cell
    auto dmap0 = dofmap.cell_dofs(cell_index0);
    auto dmap1 = dofmap.cell_dofs(cell_index1);

    // Get cell geometry
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
      const int num_entries = offsets[i + 1] - offsets[i];
      coeff_array.segment(2 * offsets[i], num_entries)
          = coeff_cell0.segment(offsets[i], num_entries);
      coeff_array.segment(offsets[i + 1] + offsets[i], num_entries)
          = coeff_cell1.segment(offsets[i], num_entries);
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
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>&
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
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>&
        bc_values1,
    const std::vector<bool>& bc_markers1,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>& x0,
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
