// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_matrix_impl.h"
#include "DofMap.h"
#include "Form.h"
#include "utils.h"
#include <dolfinx/function/Constant.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/CoordinateDofs.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshEntity.h>
#include <dolfinx/mesh/MeshIterator.h>
#include <petscsys.h>

using namespace dolfinx;

//-----------------------------------------------------------------------------
void fem::impl::assemble_matrix(Mat A, const Form& a,
                                const std::vector<bool>& bc0,
                                const std::vector<bool>& bc1)
{
  assert(a.mesh());
  const mesh::Mesh& mesh = *a.mesh();

  // Get dofmap data
  const fem::DofMap& dofmap0 = *a.function_space(0)->dofmap();
  const fem::DofMap& dofmap1 = *a.function_space(1)->dofmap();
  auto& dof_array0 = dofmap0.dof_array();
  auto& dof_array1 = dofmap1.dof_array();

  assert(dofmap0.element_dof_layout);
  assert(dofmap1.element_dof_layout);
  const int num_dofs_per_cell0 = dofmap0.element_dof_layout->num_dofs();
  const int num_dofs_per_cell1 = dofmap1.element_dof_layout->num_dofs();

  // Prepare constants
  if (!a.all_constants_set())
    throw std::runtime_error("Unset constant in Form");

  const std::vector<
      std::pair<std::string, std::shared_ptr<const function::Constant>>>
      constants = a.constants();
  std::vector<PetscScalar> constant_values;
  for (auto& constant : constants)
  {
    const std::vector<PetscScalar>& array = constant.second->value;
    constant_values.insert(constant_values.end(), array.data(),
                           array.data() + array.size());
  }

  // Prepare coefficients
  const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::RowMajor>
      coeffs = pack_coefficients(a);

  const FormIntegrals& integrals = a.integrals();
  using type = fem::FormIntegrals::Type;
  for (int i = 0; i < integrals.num_integrals(type::cell); ++i)
  {
    auto& fn = integrals.get_tabulate_tensor(type::cell, i);
    const std::vector<std::int32_t>& active_cells
        = integrals.integral_domains(type::cell, i);
    fem::impl::assemble_cells(
        A, mesh, active_cells, dof_array0, num_dofs_per_cell0, dof_array1,
        num_dofs_per_cell1, bc0, bc1, fn, coeffs, constant_values);
  }

  for (int i = 0; i < integrals.num_integrals(type::exterior_facet); ++i)
  {
    auto& fn = integrals.get_tabulate_tensor(type::exterior_facet, i);
    const std::vector<std::int32_t>& active_facets
        = integrals.integral_domains(type::exterior_facet, i);
    fem::impl::assemble_exterior_facets(A, mesh, active_facets, dofmap0,
                                        dofmap1, bc0, bc1, fn, coeffs,
                                        constant_values);
  }

  for (int i = 0; i < integrals.num_integrals(type::interior_facet); ++i)
  {
    const std::vector<int> c_offsets = a.coefficients().offsets();
    auto& fn = integrals.get_tabulate_tensor(type::interior_facet, i);
    const std::vector<std::int32_t>& active_facets
        = integrals.integral_domains(type::interior_facet, i);
    fem::impl::assemble_interior_facets(A, mesh, active_facets, dofmap0,
                                        dofmap1, bc0, bc1, fn, coeffs,
                                        c_offsets, constant_values);
  }
}
//-----------------------------------------------------------------------------
void fem::impl::assemble_cells(
    Mat A, const mesh::Mesh& mesh,
    const std::vector<std::int32_t>& active_cells,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofmap0,
    int num_dofs_per_cell0,
    const Eigen::Ref<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>>& dofmap1,
    int num_dofs_per_cell1, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const std::uint8_t*, const bool*, const bool*,
                             const std::uint8_t*)>& kernel,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const std::vector<PetscScalar>& constant_values)
{
  assert(A);
  const int gdim = mesh.geometry().dim();
  mesh.create_entity_permutations();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& connectivity_g
      = mesh.coordinate_dofs().entity_points();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& pos_g
      = connectivity_g.offsets();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& cell_g
      = connectivity_g.array();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().points();

  // Data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;

  // Iterate over active cells
  PetscErrorCode ierr;
  for (std::int32_t cell_index : active_cells)
  {
    // Get cell coordinates/geometry
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index] + i], j);

    const mesh::MeshEntity cell(mesh, gdim, cell_index);

    const Eigen::Ref<const Eigen::Array<bool, 1, Eigen::Dynamic>>
        cell_edge_reflections
        = mesh.topology().get_edge_reflections(cell_index);
    const Eigen::Ref<const Eigen::Array<bool, 1, Eigen::Dynamic>>
        cell_face_reflections
        = mesh.topology().get_face_reflections(cell_index);
    const Eigen::Ref<const Eigen::Array<std::uint8_t, 1, Eigen::Dynamic>>
        cell_face_rotations = mesh.topology().get_face_rotations(cell_index);

    // Tabulate tensor
    auto coeff_cell = coeffs.row(cell_index);
    Ae.setZero(num_dofs_per_cell0, num_dofs_per_cell1);
    kernel(Ae.data(), coeff_cell.data(), constant_values.data(),
           coordinate_dofs.data(), nullptr, nullptr,
           cell_edge_reflections.data(), cell_face_reflections.data(),
           cell_face_rotations.data());

    // Zero rows/columns for essential bcs
    if (!bc0.empty())
    {
      for (Eigen::Index i = 0; i < Ae.rows(); ++i)
      {
        const std::int32_t dof = dofmap0[cell_index * num_dofs_per_cell0 + i];
        if (bc0[dof])
          Ae.row(i).setZero();
      }
    }
    if (!bc1.empty())
    {
      for (Eigen::Index j = 0; j < Ae.cols(); ++j)
      {
        const std::int32_t dof = dofmap1[cell_index * num_dofs_per_cell1 + j];
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
    const std::vector<std::int32_t>& active_facets, const DofMap& dofmap0,
    const DofMap& dofmap1, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const std::uint8_t*, const bool*, const bool*,
                             const std::uint8_t*)>& fn,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const std::vector<PetscScalar> constant_values)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();
  mesh.create_entities(tdim - 1);
  mesh.create_connectivity(tdim - 1, tdim);
  mesh.create_entity_permutations();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& connectivity_g
      = mesh.coordinate_dofs().entity_points();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& pos_g
      = connectivity_g.offsets();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& cell_g
      = connectivity_g.array();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().points();

  // Data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;

  // Iterate over all facets
  PetscErrorCode ierr;
  auto f_to_c = mesh.topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  for (const auto& f : active_facets)
  {
    // Create attached cell
    auto cells = f_to_c->links(f);
    assert(cells.rows() == 1);
    const mesh::MeshEntity cell(mesh, tdim, cells[0]);

    // Get local index of facet with respect to the cell
    const mesh::MeshEntity facet(mesh, tdim - 1, f);
    const int local_facet = cell.index(facet);

    // Get cell vertex coordinates
    for (int i = 0; i < num_dofs_g; ++i)
      for (int j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cells[0]] + i], j);

    // Get the permutation of the facet
    const std::uint8_t perm = mesh.topology().get_facet_permutation(
        cell.index(), facet.dim(), local_facet);

    // Get dof maps for cell
    auto dmap0 = dofmap0.cell_dofs(cell.index());
    auto dmap1 = dofmap1.cell_dofs(cell.index());

    const Eigen::Ref<const Eigen::Array<bool, 1, Eigen::Dynamic>>
        cell_edge_reflections
        = mesh.topology().get_edge_reflections(cell.index());
    const Eigen::Ref<const Eigen::Array<bool, 1, Eigen::Dynamic>>
        cell_face_reflections
        = mesh.topology().get_face_reflections(cell.index());
    const Eigen::Ref<const Eigen::Array<std::uint8_t, 1, Eigen::Dynamic>>
        cell_face_rotations = mesh.topology().get_face_rotations(cell.index());

    // Tabulate tensor
    auto coeff_cell = coeffs.row(cells[0]);
    Ae.setZero(dmap0.size(), dmap1.size());
    fn(Ae.data(), coeff_cell.data(), constant_values.data(),
       coordinate_dofs.data(), &local_facet, &perm,
       cell_edge_reflections.data(), cell_face_reflections.data(),
       cell_face_rotations.data());

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
void fem::impl::assemble_interior_facets(
    Mat A, const mesh::Mesh& mesh,
    const std::vector<std::int32_t>& active_facets, const DofMap& dofmap0,
    const DofMap& dofmap1, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const std::uint8_t*, const bool*, const bool*,
                             const std::uint8_t*)>& fn,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const std::vector<int>& offsets,
    const std::vector<PetscScalar>& constant_values)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();
  mesh.create_entities(tdim - 1);
  mesh.create_connectivity(tdim - 1, tdim);
  mesh.create_entity_permutations();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& connectivity_g
      = mesh.coordinate_dofs().entity_points();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& pos_g
      = connectivity_g.offsets();
  const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& cell_g
      = connectivity_g.array();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = connectivity_g.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().points();

  // Data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(2 * num_dofs_g, gdim);
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(2 * offsets.back());
  assert(offsets.back() == coeffs.cols());

  // Temporaries for joint dofmaps
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> dmapjoint0, dmapjoint1;

  // Iterate over all facets
  PetscErrorCode ierr;
  auto c = mesh.topology().connectivity(tdim - 1, tdim);
  assert(c);
  for (const auto& facet_index : active_facets)
  {
    const mesh::MeshEntity facet(mesh, tdim - 1, facet_index);
    assert(mesh.topology().interior_facets()[facet_index]);

    // Create attached cells
    auto cells = c->links(facet_index);
    assert(cells.rows() == 2);
    const mesh::MeshEntity cell0(mesh, tdim, facet.entities(tdim)[0]);
    const mesh::MeshEntity cell1(mesh, tdim, facet.entities(tdim)[1]);

    // Get local index of facet with respect to the cell
    const std::array<int, 2> local_facet
        = {cell0.index(facet), cell1.index(facet)};

    // Get cell vertex coordinates
    const int cell_index0 = cell0.index();
    const int cell_index1 = cell1.index();

    const std::array<std::uint8_t, 2> perm
        = {mesh.topology().get_facet_permutation(cell_index0, facet.dim(),
                                                 local_facet[0]),
           mesh.topology().get_facet_permutation(cell_index1, facet.dim(),
                                                 local_facet[1])};

    for (int i = 0; i < num_dofs_g; ++i)
    {
      for (int j = 0; j < gdim; ++j)
      {
        coordinate_dofs(i, j) = x_g(cell_g[pos_g[cell_index0] + i], j);
        coordinate_dofs(i + num_dofs_g, j)
            = x_g(cell_g[pos_g[cell_index1] + i], j);
      }
    }

    // Get dof maps for cells and pack
    auto dmap0_cell0 = dofmap0.cell_dofs(cell_index0);
    auto dmap0_cell1 = dofmap0.cell_dofs(cell_index1);
    dmapjoint0.resize(dmap0_cell0.size() + dmap0_cell1.size());
    dmapjoint0.head(dmap0_cell0.size()) = dmap0_cell0;
    dmapjoint0.tail(dmap0_cell1.size()) = dmap0_cell1;

    auto dmap1_cell0 = dofmap1.cell_dofs(cell_index0);
    auto dmap1_cell1 = dofmap1.cell_dofs(cell_index1);
    dmapjoint1.resize(dmap1_cell0.size() + dmap1_cell1.size());
    dmapjoint1.head(dmap1_cell0.size()) = dmap1_cell0;
    dmapjoint1.tail(dmap1_cell1.size()) = dmap1_cell1;

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

    const Eigen::Ref<const Eigen::Array<bool, 1, Eigen::Dynamic>>
        cell_edge_reflections
        = mesh.topology().get_edge_reflections(cell_index0);
    const Eigen::Ref<const Eigen::Array<bool, 1, Eigen::Dynamic>>
        cell_face_reflections
        = mesh.topology().get_face_reflections(cell_index0);
    const Eigen::Ref<const Eigen::Array<std::uint8_t, 1, Eigen::Dynamic>>
        cell_face_rotations = mesh.topology().get_face_rotations(cell_index0);

    // Tabulate tensor
    Ae.setZero(dmapjoint0.size(), dmapjoint1.size());
    fn(Ae.data(), coeff_array.data(), constant_values.data(),
       coordinate_dofs.data(), local_facet.data(), perm.data(),
       cell_edge_reflections.data(), cell_face_reflections.data(),
       cell_face_rotations.data());

    // Zero rows/columns for essential bcs
    if (!bc0.empty())
    {
      for (Eigen::Index i = 0; i < dmapjoint0.size(); ++i)
      {
        if (bc0[dmapjoint0[i]])
          Ae.row(i).setZero();
      }
    }
    if (!bc1.empty())
    {
      for (Eigen::Index j = 0; j < dmapjoint1.size(); ++j)
      {
        if (bc1[dmapjoint1[j]])
          Ae.col(j).setZero();
      }
    }

    ierr = MatSetValuesLocal(A, dmapjoint0.size(), dmapjoint0.data(),
                             dmapjoint1.size(), dmapjoint1.data(), Ae.data(),
                             ADD_VALUES);
#ifdef DEBUG
    if (ierr != 0)
      la::petsc_error(ierr, __FILE__, "MatSetValuesLocal");
#endif
  }
}
//-----------------------------------------------------------------------------
