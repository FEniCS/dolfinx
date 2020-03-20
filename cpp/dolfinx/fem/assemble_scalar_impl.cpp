// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_scalar_impl.h"
#include "DofMap.h"
#include "Form.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/function/Constant.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <petscsys.h>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
PetscScalar dolfinx::fem::impl::assemble_scalar(const dolfinx::fem::Form& M)
{
  assert(M.mesh());
  const mesh::Mesh& mesh = *M.mesh();

  // Prepare constants
  if (!M.all_constants_set())
    throw std::runtime_error("Unset constant in Form");
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
    auto& fn = integrals.get_tabulate_tensor(type::cell, i);
    const std::vector<std::int32_t>& active_cells
        = integrals.integral_domains(type::cell, i);
    value += fem::impl::assemble_cells(mesh, active_cells, fn, coeffs,
                                       constant_values);
  }

  for (int i = 0; i < integrals.num_integrals(type::exterior_facet); ++i)
  {
    auto& fn = integrals.get_tabulate_tensor(type::exterior_facet, i);
    const std::vector<std::int32_t>& active_facets
        = integrals.integral_domains(type::exterior_facet, i);
    value += fem::impl::assemble_exterior_facets(mesh, active_facets, fn,
                                                 coeffs, constant_values);
  }

  for (int i = 0; i < integrals.num_integrals(type::interior_facet); ++i)
  {
    const std::vector<int> c_offsets = M.coefficients().offsets();
    auto& fn = integrals.get_tabulate_tensor(type::interior_facet, i);
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
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const std::vector<PetscScalar>& constant_values)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();
  mesh.create_entities(tdim);
  mesh.create_entity_permutations();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().x();

  // Create data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);

  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = mesh.topology().get_cell_permutation_info();

  // Iterate over all cells
  PetscScalar value(0);
  for (std::int32_t c : active_cells)
  {
    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

    auto coeff_cell = coeffs.row(c);
    fn(&value, coeff_cell.data(), constant_values.data(),
       coordinate_dofs.data(), nullptr, nullptr, cell_info[c]);
  }

  return value;
}
//-----------------------------------------------------------------------------
PetscScalar fem::impl::assemble_exterior_facets(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const std::vector<PetscScalar>& constant_values)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();
  mesh.create_entities(tdim - 1);
  mesh.create_connectivity(tdim - 1, tdim);
  mesh.create_entity_permutations();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().x();

  // Creat data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);

  const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms
      = mesh.topology().get_facet_permutations();
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = mesh.topology().get_cell_permutation_info();

  auto f_to_c = mesh.topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);

  // Iterate over all facets
  PetscScalar value(0);
  for (std::int32_t facet : active_facets)
  {
    // Create attached cell
    assert(f_to_c->num_links(facet) == 1);
    const int cell = f_to_c->links(facet)[0];

    // Get local index of facet with respect to the cell
    auto facets = c_to_f->links(cell);
    auto it = std::find(facets.data(), facets.data() + facets.rows(), facet);
    assert(it != (facets.data() + facets.rows()));
    const int local_facet = std::distance(facets.data(), it);

    auto x_dofs = x_dofmap.links(cell);
    for (int i = 0; i < num_dofs_g; ++i)
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

    auto coeff_cell = coeffs.row(cell);
    const std::uint8_t perm = perms(local_facet, cell);
    fn(&value, coeff_cell.data(), constant_values.data(),
       coordinate_dofs.data(), &local_facet, &perm, cell_info[cell]);
  }

  return value;
}
//-----------------------------------------------------------------------------
PetscScalar fem::impl::assemble_interior_facets(
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
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
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().x();

  // Creat data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(2 * num_dofs_g, gdim);
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(2 * offsets.back());
  assert(offsets.back() == coeffs.cols());

  const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms
      = mesh.topology().get_facet_permutations();
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = mesh.topology().get_cell_permutation_info();

  auto f_to_c = mesh.topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);

  // Iterate over all facets
  PetscScalar value(0);
  for (std::int32_t f : active_facets)
  {
    // Create attached cell
    auto cells = f_to_c->links(f);
    assert(cells.rows() == 2);

    // Get local index of facet with respect to the cell
    std::array<int, 2> local_facet;
    for (int i = 0; i < 2; ++i)
    {
      auto facets = c_to_f->links(cells[i]);
      auto it = std::find(facets.data(), facets.data() + facets.rows(), f);
      assert(it != (facets.data() + facets.rows()));
      local_facet[i] = std::distance(facets.data(), it);
    }

    const std::array<std::uint8_t, 2> perm
        = {perms(local_facet[0], cells[0]), perms(local_facet[1], cells[1])};

    // Get cell geometry
    auto x_dofs0 = x_dofmap.links(cells[0]);
    auto x_dofs1 = x_dofmap.links(cells[1]);
    for (int i = 0; i < num_dofs_g; ++i)
    {
      coordinate_dofs.row(i) = x_g.row(x_dofs0[i]).head(gdim);
      coordinate_dofs.row(i + num_dofs_g) = x_g.row(x_dofs1[i]).head(gdim);
    }

    // Layout for the restricted coefficients is flattened
    // w[coefficient][restriction][dof]
    auto coeff_cell0 = coeffs.row(cells[0]);
    auto coeff_cell1 = coeffs.row(cells[1]);

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
    fn(&value, coeff_array.data(), constant_values.data(),
       coordinate_dofs.data(), local_facet.data(), perm.data(),
       cell_info[cells[0]]);
  }

  return value;
}
//-----------------------------------------------------------------------------
