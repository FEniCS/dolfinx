// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_matrix_impl.h"
#include "DofMap.h"
#include "Form.h"
#include "utils.h"
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <petscsys.h>

using namespace dolfinx;

//-----------------------------------------------------------------------------
template <typename ScalarType>
void fem::impl::assemble_matrix(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const ScalarType*)>&
        mat_set_values_local,
    const Form& a, const std::vector<bool>& bc0, const std::vector<bool>& bc1)
{
  std::shared_ptr<const mesh::Mesh> mesh = a.mesh();
  assert(mesh);

  // Get dofmap data
  std::shared_ptr<const fem::DofMap> dofmap0 = a.function_space(0)->dofmap();
  std::shared_ptr<const fem::DofMap> dofmap1 = a.function_space(1)->dofmap();
  assert(dofmap0);
  assert(dofmap1);
  const graph::AdjacencyList<std::int32_t>& dofs0 = dofmap0->list();
  const graph::AdjacencyList<std::int32_t>& dofs1 = dofmap1->list();

  assert(dofmap0->element_dof_layout);
  assert(dofmap1->element_dof_layout);
  const int num_dofs_per_cell0 = dofmap0->element_dof_layout->num_dofs();
  const int num_dofs_per_cell1 = dofmap1->element_dof_layout->num_dofs();

  // Prepare constants
  if (!a.all_constants_set())
    throw std::runtime_error("Unset constant in Form");
  const Eigen::Array<ScalarType, Eigen::Dynamic, 1> constant_values
      = pack_constants(a);

  // Prepare coefficients
  const Eigen::Array<ScalarType, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::RowMajor>
      coeffs = pack_coefficients(a);

  const FormIntegrals& integrals = a.integrals();
  using type = fem::FormIntegrals::Type;
  for (int i = 0; i < integrals.num_integrals(type::cell); ++i)
  {
    const auto& fn = integrals.get_tabulate_tensor(type::cell, i);
    const std::vector<std::int32_t>& active_cells
        = integrals.integral_domains(type::cell, i);

    fem::impl::assemble_cells<ScalarType>(
        mat_set_values_local, *mesh, active_cells, dofs0, num_dofs_per_cell0,
        dofs1, num_dofs_per_cell1, bc0, bc1, fn, coeffs, constant_values);
  }

  for (int i = 0; i < integrals.num_integrals(type::exterior_facet); ++i)
  {
    const auto& fn = integrals.get_tabulate_tensor(type::exterior_facet, i);
    const std::vector<std::int32_t>& active_facets
        = integrals.integral_domains(type::exterior_facet, i);
    fem::impl::assemble_exterior_facets<ScalarType>(
        mat_set_values_local, *mesh, active_facets, *dofmap0, *dofmap1, bc0,
        bc1, fn, coeffs, constant_values);
  }

  for (int i = 0; i < integrals.num_integrals(type::interior_facet); ++i)
  {
    const std::vector<int> c_offsets = a.coefficients().offsets();
    const auto& fn = integrals.get_tabulate_tensor(type::interior_facet, i);
    const std::vector<std::int32_t>& active_facets
        = integrals.integral_domains(type::interior_facet, i);
    fem::impl::assemble_interior_facets<ScalarType>(
        mat_set_values_local, *mesh, active_facets, *dofmap0, *dofmap1, bc0,
        bc1, fn, coeffs, c_offsets, constant_values);
  }
}
//-----------------------------------------------------------------------------
// @cond
// protect from Doxygen
// Explicit instantiation with PetscScalar
template void fem::impl::assemble_matrix<PetscScalar>(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const PetscScalar*)>&
        mat_set_values_local,
    const Form& a, const std::vector<bool>& bc0, const std::vector<bool>& bc1);
// @endcond
//-----------------------------------------------------------------------------
template <typename ScalarType>
void fem::impl::assemble_cells(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const ScalarType*)>&
        mat_set_values_local,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_cells,
    const graph::AdjacencyList<std::int32_t>& dofmap0, int num_dofs_per_cell0,
    const graph::AdjacencyList<std::int32_t>& dofmap1, int num_dofs_per_cell1,
    const std::vector<bool>& bc0, const std::vector<bool>& bc1,
    const std::function<void(ScalarType*, const ScalarType*, const ScalarType*,
                             const double*, const int*, const std::uint8_t*,
                             const std::uint32_t)>& kernel,
    const Eigen::Array<ScalarType, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const Eigen::Array<ScalarType, Eigen::Dynamic, 1>& constant_values)
{
  const int gdim = mesh.geometry().dim();
  mesh.topology_mutable().create_entity_permutations();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().x();

  // Data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ae;

  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = mesh.topology().get_cell_permutation_info();

  // Iterate over active cells
  for (std::int32_t c : active_cells)
  {
    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < x_dofs.rows(); ++i)
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

    // Tabulate tensor
    auto coeff_cell = coeffs.row(c);
    Ae.setZero(num_dofs_per_cell0, num_dofs_per_cell1);
    kernel(Ae.data(), coeff_cell.data(), constant_values.data(),
           coordinate_dofs.data(), nullptr, nullptr, cell_info[c]);

    auto dofs0 = dofmap0.links(c);
    auto dofs1 = dofmap1.links(c);

    // Zero rows/columns for essential bcs
    if (!bc0.empty())
    {
      for (Eigen::Index i = 0; i < Ae.rows(); ++i)
      {
        const std::int32_t dof = dofs0[i];
        if (bc0[dof])
          Ae.row(i).setZero();
      }
    }
    if (!bc1.empty())
    {
      for (Eigen::Index j = 0; j < Ae.cols(); ++j)
      {
        const std::int32_t dof = dofs1[j];
        if (bc1[dof])
          Ae.col(j).setZero();
      }
    }

    mat_set_values_local(num_dofs_per_cell0, dofs0.data(), num_dofs_per_cell1,
                         dofs1.data(), Ae.data());
  }
}
//-----------------------------------------------------------------------------
template <typename ScalarType>
void fem::impl::assemble_exterior_facets(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const ScalarType*)>&
        mat_set_values_local,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const DofMap& dofmap0, const DofMap& dofmap1, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1,
    const std::function<void(ScalarType*, const ScalarType*, const ScalarType*,
                             const double*, const int*, const std::uint8_t*,
                             const std::uint32_t)>& kernel,
    const Eigen::Array<ScalarType, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const Eigen::Array<ScalarType, Eigen::Dynamic, 1> constant_values)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();

  // FIXME: cleanup these calls? Some of the happen internally again.
  mesh.topology_mutable().create_entities(tdim - 1);
  mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
  mesh.topology_mutable().create_entity_permutations();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().x();

  // Data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ae;

  const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms
      = mesh.topology().get_facet_permutations();
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = mesh.topology().get_cell_permutation_info();

  // Iterate over all facets
  auto f_to_c = mesh.topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);
  for (std::int32_t f : active_facets)
  {
    auto cells = f_to_c->links(f);
    assert(cells.rows() == 1);

    // Get local index of facet with respect to the cell
    auto facets = c_to_f->links(cells[0]);
    const auto* it = std::find(facets.data(), facets.data() + facets.rows(), f);
    assert(it != (facets.data() + facets.rows()));
    const int local_facet = std::distance(facets.data(), it);

    // Get cell vertex coordinates
    auto x_dofs = x_dofmap.links(cells[0]);
    for (int i = 0; i < num_dofs_g; ++i)
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

    // Get dof maps for cell
    auto dmap0 = dofmap0.cell_dofs(cells[0]);
    auto dmap1 = dofmap1.cell_dofs(cells[0]);

    // Tabulate tensor
    auto coeff_cell = coeffs.row(cells[0]);
    const std::uint8_t perm = perms(local_facet, cells[0]);
    Ae.setZero(dmap0.size(), dmap1.size());
    kernel(Ae.data(), coeff_cell.data(), constant_values.data(),
           coordinate_dofs.data(), &local_facet, &perm, cell_info[cells[0]]);

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

    mat_set_values_local(dmap0.size(), dmap0.data(), dmap1.size(), dmap1.data(),
                         Ae.data());
  }
}
//-----------------------------------------------------------------------------
template <typename ScalarType>
void fem::impl::assemble_interior_facets(
    const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                            const std::int32_t*, const ScalarType*)>&
        mat_set_values_local,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const DofMap& dofmap0, const DofMap& dofmap1, const std::vector<bool>& bc0,
    const std::vector<bool>& bc1,
    const std::function<void(ScalarType*, const ScalarType*, const ScalarType*,
                             const double*, const int*, const std::uint8_t*,
                             const std::uint32_t)>& fn,
    const Eigen::Array<ScalarType, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const std::vector<int>& offsets,
    const Eigen::Array<ScalarType, Eigen::Dynamic, 1>& constant_values)
{
  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();

  // FIXME: cleanup these calls? Some of the happen internally again.
  mesh.topology_mutable().create_entities(tdim - 1);
  mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
  mesh.topology_mutable().create_entity_permutations();

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);

  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().x();

  // Data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(2 * num_dofs_g, gdim);
  Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ae;
  Eigen::Array<ScalarType, Eigen::Dynamic, 1> coeff_array(2 * offsets.back());
  assert(offsets.back() == coeffs.cols());

  // Temporaries for joint dofmaps
  Eigen::Array<std::int32_t, Eigen::Dynamic, 1> dmapjoint0, dmapjoint1;

  const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms
      = mesh.topology().get_facet_permutations();
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = mesh.topology().get_cell_permutation_info();

  // Iterate over all facets
  auto c = mesh.topology().connectivity(tdim - 1, tdim);
  assert(c);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);
  for (std::int32_t facet_index : active_facets)
  {
    // Create attached cells
    auto cells = c->links(facet_index);
    assert(cells.rows() == 2);

    // Get local index of facet with respect to the cell
    auto facets0 = c_to_f->links(cells[0]);
    const auto* it0 = std::find(facets0.data(), facets0.data() + facets0.rows(),
                                facet_index);
    assert(it0 != (facets0.data() + facets0.rows()));
    const int local_facet0 = std::distance(facets0.data(), it0);
    auto facets1 = c_to_f->links(cells[1]);
    const auto* it1 = std::find(facets1.data(), facets1.data() + facets1.rows(),
                                facet_index);
    assert(it1 != (facets1.data() + facets1.rows()));
    const int local_facet1 = std::distance(facets1.data(), it1);

    const std::array<int, 2> local_facet = {local_facet0, local_facet1};

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

    // Get dof maps for cells and pack
    auto dmap0_cell0 = dofmap0.cell_dofs(cells[0]);
    auto dmap0_cell1 = dofmap0.cell_dofs(cells[1]);
    dmapjoint0.resize(dmap0_cell0.size() + dmap0_cell1.size());
    dmapjoint0.head(dmap0_cell0.size()) = dmap0_cell0;
    dmapjoint0.tail(dmap0_cell1.size()) = dmap0_cell1;

    auto dmap1_cell0 = dofmap1.cell_dofs(cells[0]);
    auto dmap1_cell1 = dofmap1.cell_dofs(cells[1]);
    dmapjoint1.resize(dmap1_cell0.size() + dmap1_cell1.size());
    dmapjoint1.head(dmap1_cell0.size()) = dmap1_cell0;
    dmapjoint1.tail(dmap1_cell1.size()) = dmap1_cell1;

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

    // Tabulate tensor
    Ae.setZero(dmapjoint0.size(), dmapjoint1.size());
    fn(Ae.data(), coeff_array.data(), constant_values.data(),
       coordinate_dofs.data(), local_facet.data(), perm.data(),
       cell_info[cells[0]]);

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

    mat_set_values_local(dmapjoint0.size(), dmapjoint0.data(),
                         dmapjoint1.size(), dmapjoint1.data(), Ae.data());
  }
}
//-----------------------------------------------------------------------------
