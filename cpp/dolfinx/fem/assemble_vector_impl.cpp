// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_vector_impl.h"
#include "DirichletBC.h"
#include "DofMap.h"
#include "Form.h"
#include "utils.h"
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/function/Constant.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <petscsys.h>

using namespace dolfinx;
using namespace dolfinx::fem;

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

  mesh.create_entity_permutations();

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
                           const double*, const int*, const std::uint8_t*,
                           const std::uint32_t)>& fn
      = a.integrals().get_tabulate_tensor(FormIntegrals::Type::cell, 0);

  // Prepare cell geometry
  const int gdim = mesh.geometry().dim();
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();

  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().x();

  // Data structures used in bc application
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;

  // Prepare constants
  if (!a.all_constants_set())
    throw std::runtime_error("Unset constant in Form");
  const Eigen::Array<PetscScalar, Eigen::Dynamic, 1> constant_values
      = pack_constants(a);

  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = mesh.topology().get_cell_permutation_info();

  // Iterate over all cells
  const int tdim = mesh.topology().dim();
  auto map = mesh.topology().index_map(tdim);
  assert(map);
  const int num_cells = map->size_local() + map->num_ghosts();
  for (int c = 0; c < num_cells; ++c)
  {
    // Get dof maps for cell
    auto dmap1 = dofmap1.cell_dofs(c);

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
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

    // Size data structure for assembly
    auto dmap0 = dofmap0.cell_dofs(c);

    auto coeff_array = coeffs.row(c);
    Ae.setZero(dmap0.size(), dmap1.size());
    fn(Ae.data(), coeff_array.data(), constant_values.data(),
       coordinate_dofs.data(), nullptr, nullptr, cell_info[c]);

    // Size data structure for assembly
    be.setZero(dmap0.size());
    for (Eigen::Index j = 0; j < dmap1.size(); ++j)
    {
      const std::int32_t jj = dmap1[j];
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

  mesh.create_entity_permutations();

  const int gdim = mesh.geometry().dim();
  const int tdim = mesh.topology().dim();
  mesh.create_entities(tdim - 1);
  mesh.create_connectivity(tdim - 1, tdim);
  mesh.create_entity_permutations();

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
                           const double*, const int*, const std::uint8_t*,
                           const std::uint32_t)>& fn
      = a.integrals().get_tabulate_tensor(FormIntegrals::Type::exterior_facet,
                                          0);

  // Prepare cell geometry
  const graph::AdjacencyList<std::int32_t>& x_dofmap = mesh.geometry().dofmap();
  // FIXME: Add proper interface for num coordinate dofs
  const int num_dofs_g = x_dofmap.num_links(0);
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x_g
      = mesh.geometry().x();

  // Data structures used in bc application
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;

  // Prepare constants
  if (!a.all_constants_set())
    throw std::runtime_error("Unset constant in Form");
  const Eigen::Array<PetscScalar, Eigen::Dynamic, 1> constant_values
      = pack_constants(a);

  // Iterate over owned facets
  const mesh::Topology& topology = mesh.topology();
  auto connectivity = topology.connectivity(tdim - 1, tdim);
  assert(connectivity);
  auto c_to_f = topology.connectivity(tdim, tdim - 1);
  assert(c_to_f);
  auto map = topology.index_map(tdim - 1);
  assert(map);

  const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms
      = mesh.topology().get_facet_permutations();
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = mesh.topology().get_cell_permutation_info();

  for (int f = 0; f < map->size_local(); ++f)
  {
    // Move to next facet if this one is an interior facet
    if (topology.interior_facets()[f])
      continue;

    // Create attached cell
    assert(connectivity->num_links(f) == 1);
    const std::int32_t cell = connectivity->links(f)[0];

    // Get local index of facet with respect to the cell
    auto facets = c_to_f->links(cell);
    const auto *it = std::find(facets.data(), facets.data() + facets.rows(), f);
    assert(it != (facets.data() + facets.rows()));
    const int local_facet = std::distance(facets.data(), it);

    const std::uint8_t perm = perms(local_facet, cell);

    // Get dof maps for cell
    auto dmap1 = dofmap1.cell_dofs(cell);

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
    auto x_dofs = x_dofmap.links(cell);
    for (int i = 0; i < num_dofs_g; ++i)
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

    // Size data structure for assembly
    auto dmap0 = dofmap0.cell_dofs(cell);

    // TODO: Move gathering of coefficients outside of main assembly
    // loop

    auto coeff_array = coeffs.row(cell);
    Ae.setZero(dmap0.size(), dmap1.size());
    fn(Ae.data(), coeff_array.data(), constant_values.data(),
       coordinate_dofs.data(), &local_facet, &perm, cell_info[cell]);

    // Size data structure for assembly
    be.setZero(dmap0.size());
    for (Eigen::Index j = 0; j < dmap1.size(); ++j)
    {
      const std::int32_t jj = dmap1[j];
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
  const graph::AdjacencyList<PetscInt>& dofs = dofmap.list();

  assert(dofmap.element_dof_layout);
  const int num_dofs_per_cell = dofmap.element_dof_layout->num_dofs();

  // Prepare constants
  if (!L.all_constants_set())
    throw std::runtime_error("Unset constant in Form");
  const Eigen::Array<PetscScalar, Eigen::Dynamic, 1> constant_values
      = pack_constants(L);

  // Prepare coefficients
  const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                     Eigen::RowMajor>
      coeffs = pack_coefficients(L);

  const FormIntegrals& integrals = L.integrals();
  using type = fem::FormIntegrals::Type;
  for (int i = 0; i < integrals.num_integrals(type::cell); ++i)
  {
    const auto & fn = integrals.get_tabulate_tensor(FormIntegrals::Type::cell, i);
    const std::vector<std::int32_t>& active_cells
        = integrals.integral_domains(type::cell, i);
    fem::impl::assemble_cells(b, mesh, active_cells, dofs, num_dofs_per_cell,
                              fn, coeffs, constant_values);
  }

  for (int i = 0; i < integrals.num_integrals(type::exterior_facet); ++i)
  {
    const auto& fn
        = integrals.get_tabulate_tensor(FormIntegrals::Type::exterior_facet, i);
    const std::vector<std::int32_t>& active_facets
        = integrals.integral_domains(type::exterior_facet, i);
    fem::impl::assemble_exterior_facets(b, mesh, active_facets, dofmap, fn,
                                        coeffs, constant_values);
  }

  for (int i = 0; i < integrals.num_integrals(type::interior_facet); ++i)
  {
    const std::vector<int> c_offsets = L.coefficients().offsets();
    const auto& fn
        = integrals.get_tabulate_tensor(FormIntegrals::Type::interior_facet, i);
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
    const graph::AdjacencyList<PetscInt>& dofmap, int num_dofs_per_cell,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& kernel,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>& constant_values)
{
  const int gdim = mesh.geometry().dim();

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
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be(num_dofs_per_cell);

  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = mesh.topology().get_cell_permutation_info();

  // Iterate over active cells
  for (std::int32_t c : active_cells)
  {
    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(c);
    for (int i = 0; i < num_dofs_g; ++i)
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

    // Tabulate vector for cell
    auto coeff_cell = coeffs.row(c);
    be.setZero();
    kernel(be.data(), coeff_cell.data(), constant_values.data(),
           coordinate_dofs.data(), nullptr, nullptr, cell_info[c]);

    // Scatter cell vector to 'global' vector array
    auto dofs = dofmap.links(c);
    for (Eigen::Index i = 0; i < num_dofs_per_cell; ++i)
      b[dofs[i]] += be[i];
  }
}
//-----------------------------------------------------------------------------
void fem::impl::assemble_exterior_facets(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const mesh::Mesh& mesh, const std::vector<std::int32_t>& active_facets,
    const fem::DofMap& dofmap,
    const std::function<void(PetscScalar*, const PetscScalar*,
                             const PetscScalar*, const double*, const int*,
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>& constant_values)
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

  // Create data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs(num_dofs_g, gdim);
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;

  const Eigen::Array<std::uint8_t, Eigen::Dynamic, Eigen::Dynamic>& perms
      = mesh.topology().get_facet_permutations();
  const Eigen::Array<std::uint32_t, Eigen::Dynamic, 1>& cell_info
      = mesh.topology().get_cell_permutation_info();

  auto f_to_c = mesh.topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);
  for (const auto& f : active_facets)
  {
    // Get index of first attached cell
    assert(f_to_c->num_links(f) > 0);
    const std::int32_t cell = f_to_c->links(f)[0];

    // Get local index of facet with respect to the cell
    auto facets = c_to_f->links(cell);
    const auto *it = std::find(facets.data(), facets.data() + facets.rows(), f);
    assert(it != (facets.data() + facets.rows()));
    const int local_facet = std::distance(facets.data(), it);
    const std::uint8_t perm = perms(local_facet, cell);

    // Get cell vertex coordinates
    auto x_dofs = x_dofmap.links(cell);
    for (int i = 0; i < num_dofs_g; ++i)
      coordinate_dofs.row(i) = x_g.row(x_dofs[i]).head(gdim);

    // Get dof map for cell
    auto dmap = dofmap.cell_dofs(cell);

    // Tabulate element vector
    auto coeff_cell = coeffs.row(cell);
    be.setZero(dmap.size());
    fn(be.data(), coeff_cell.data(), constant_values.data(),
       coordinate_dofs.data(), &local_facet, &perm, cell_info[cell]);

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
                             const std::uint8_t*, const std::uint32_t)>& fn,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>& coeffs,
    const std::vector<int>& offsets,
    const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>& constant_values)
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
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;
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
  for (const auto& f : active_facets)
  {
    // Get attached cell indices
    auto cells = f_to_c->links(f);
    assert(cells.rows() == 2);

    // Create attached cells
    std::array<int, 2> local_facet;
    for (int i = 0; i < 2; ++i)
    {
      auto facets = c_to_f->links(cells[i]);
      const auto *it = std::find(facets.data(), facets.data() + facets.rows(), f);
      assert(it != (facets.data() + facets.rows()));
      local_facet[i] = std::distance(facets.data(), it);
    }

    // Orientation
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

    // Get dofmaps for cell
    auto dmap0 = dofmap.cell_dofs(cells[0]);
    auto dmap1 = dofmap.cell_dofs(cells[1]);

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

    // Tabulate element vector
    be.setZero(dmap0.size() + dmap1.size());
    fn(be.data(), coeff_array.data(), constant_values.data(),
       coordinate_dofs.data(), local_facet.data(), perm.data(),
       cell_info[cells[0]]);

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
    const std::vector<std::vector<std::shared_ptr<const DirichletBC>>>& bcs1,
    const std::vector<
        Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>>>& x0,
    double scale)
{
  // FIXME: make changes to reactivate this check
  if (!x0.empty() and x0.size() != a.size())
  {
    throw std::runtime_error(
        "Mismatch in size between x0 and bilinear form in assembler.");
  }

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
          = map1->block_size() * (map1->size_local() + map1->num_ghosts());
      bc_markers1.assign(crange, false);
      bc_values1 = Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>::Zero(crange);
      for (const std::shared_ptr<const DirichletBC>& bc : bcs1[j])
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
  if (a.integrals().num_integrals(fem::FormIntegrals::Type::cell) > 0)
    _lift_bc_cells(b, a, bc_values1, bc_markers1, x0, scale);
  if (a.integrals().num_integrals(fem::FormIntegrals::Type::exterior_facet) > 0)
    _lift_bc_exterior_facets(b, a, bc_values1, bc_markers1, x0, scale);
}
//-----------------------------------------------------------------------------
