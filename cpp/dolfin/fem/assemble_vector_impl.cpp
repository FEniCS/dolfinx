// Copyright (C) 2018-2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "assemble_vector_impl.h"
#include "DirichletBC.h"
#include "Form.h"
#include "GenericDofMap.h"
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

namespace
{
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
  const fem::GenericDofMap& dofmap0 = *a.function_space(0)->dofmap();
  const fem::GenericDofMap& dofmap1 = *a.function_space(1)->dofmap();

  // TODO: simplify and move elsewhere
  // Manage coefficients
  const FormCoefficients& coefficients = a.coeffs();
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

  const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                           int)>& fn
      = a.integrals().get_tabulate_tensor_fn_cell(0);

  // Data structures used in bc application
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs;

  // Iterate over all cells
  for (const mesh::Cell& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Check that cell is not a ghost
    assert(!cell.is_ghost());

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
    cell.get_coordinate_dofs(coordinate_dofs);

    // Size data structure for assembly
    const Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap0
        = dofmap0.cell_dofs(cell.index());

    // TODO: Move gathering of coefficients outside of main assembly
    // loop
    // Update coefficients
    for (int i = 0; i < coefficients.size(); ++i)
    {
      coefficients_ptr[i]->restrict(coeff_array.data() + n[i], cell,
                                    coordinate_dofs);
    }

    Ae.setZero(dmap0.size(), dmap1.size());
    fn(Ae.data(), coeff_array.data(), coordinate_dofs.data(), 1);

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

  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim - 1);
  mesh.init(tdim - 1, tdim);

  // Get dofmap for columns and rows of a
  assert(a.function_space(0));
  assert(a.function_space(0)->dofmap());
  assert(a.function_space(1));
  assert(a.function_space(1)->dofmap());
  const fem::GenericDofMap& dofmap0 = *a.function_space(0)->dofmap();
  const fem::GenericDofMap& dofmap1 = *a.function_space(1)->dofmap();

  // TODO: simplify and move elsewhere
  // Manage coefficients
  const FormCoefficients& coefficients = a.coeffs();
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

  const std::function<void(PetscScalar*, const PetscScalar*, const double*, int,
                           int)>& fn
      = a.integrals().get_tabulate_tensor_fn_exterior_facet(0);

  // Data structures used in bc application
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs;

  // Iterate over all cells
  for (const mesh::Facet& facet : mesh::MeshRange<mesh::Facet>(mesh))
  {
    if (facet.num_global_entities(tdim) != 1)
      continue;

    // FIXME: sort out ghosts

    // Create attached cell
    mesh::Cell cell(mesh, facet.entities(tdim)[0]);

    // Get local index of facet with respect to the cell
    const int local_facet = cell.index(facet);

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
    cell.get_coordinate_dofs(coordinate_dofs);

    // Size data structure for assembly
    const Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap0
        = dofmap0.cell_dofs(cell.index());

    // TODO: Move gathering of coefficients outside of main assembly
    // loop
    // Update coefficients
    for (int i = 0; i < coefficients.size(); ++i)
    {
      coefficients_ptr[i]->restrict(coeff_array.data() + n[i], cell,
                                    coordinate_dofs);
    }

    Ae.setZero(dmap0.size(), dmap1.size());
    fn(Ae.data(), coeff_array.data(), coordinate_dofs.data(), local_facet, 1);

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
void fem::impl::assemble(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& L)
{
  assert(L.mesh());
  const mesh::Mesh& mesh = *L.mesh();
  const fem::GenericDofMap& dofmap = *L.function_space(0)->dofmap();

  // Prepare coefficients
  const FormCoefficients& coefficients = L.coeffs();
  std::vector<const function::Function*> coeff_fn(coefficients.size());
  for (int i = 0; i < coefficients.size(); ++i)
    coeff_fn[i] = coefficients.get(i).get();
  std::vector<int> c_offsets = coefficients.offsets();

  if (L.integrals().num_integrals(fem::FormIntegrals::Type::cell) > 0)
  {
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int)>& fn
        = L.integrals().get_tabulate_tensor_fn_cell(0);
    fem::impl::assemble_cells(b, mesh, dofmap, fn, coeff_fn, c_offsets);
  }

  if (L.integrals().num_integrals(fem::FormIntegrals::Type::exterior_facet) > 0)
  {
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int, int)>& fn
        = L.integrals().get_tabulate_tensor_fn_exterior_facet(0);
    fem::impl::assemble_exterior_facets(b, mesh, dofmap, fn, coeff_fn,
                                        c_offsets);
  }

  if (L.integrals().num_integrals(fem::FormIntegrals::Type::exterior_facet) > 1)
  {
    throw std::runtime_error(
        "Multiple exterior facet integrals not supported yet.");
  }

  if (L.integrals().num_integrals(fem::FormIntegrals::Type::interior_facet) > 0)
    fem::impl::assemble_interior_facets(b, L);
}
//-----------------------------------------------------------------------------
void fem::impl::assemble_cells(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const mesh::Mesh& mesh, const fem::GenericDofMap& dofmap,
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int)>& fn,
    std::vector<const function::Function*> coefficients,
    const std::vector<int>& offsets)
{
  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim);

  // Creat data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(offsets.back());

  // Iterate over all cells
  for (const mesh::Cell& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Check that cell is not a ghost
    assert(!cell.is_ghost());

    // Get cell vertex coordinates
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get dof maps for cell
    const Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap
        = dofmap.cell_dofs(cell.index());

    // Size data structure for assembly
    be.setZero(dmap.size());

    // Update coefficients and tabulate vector
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
      coefficients[i]->restrict(coeff_array.data() + offsets[i], cell,
                                coordinate_dofs);
    }
    fn(be.data(), coeff_array.data(), coordinate_dofs.data(), 1);

    // Add local cell vector to global vector
    for (Eigen::Index i = 0; i < dmap.size(); ++i)
      b[dmap[i]] += be[i];
  }
}
//-----------------------------------------------------------------------------
void fem::impl::assemble_exterior_facets(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b,
    const mesh::Mesh& mesh, const fem::GenericDofMap& dofmap,
    const std::function<void(PetscScalar*, const PetscScalar*, const double*,
                             int, int)>& fn,
    std::vector<const function::Function*> coefficients,
    const std::vector<int>& offsets)
{
  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim - 1);
  mesh.init(tdim - 1, tdim);

  // Creat data structures used in assembly
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coordinate_dofs;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> coeff_array(offsets.back());

  // Iterate over all facets
  for (const mesh::Facet& facet : mesh::MeshRange<mesh::Facet>(mesh))
  {
    if (facet.num_global_entities(tdim) != 1)
      continue;

    // TODO: check ghosting sanity?

    // Create attached cell
    mesh::Cell cell(mesh, facet.entities(tdim)[0]);

    // Get local index of facet with respect to the cell
    const int local_facet = cell.index(facet);

    // Get cell vertex coordinates
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get dof map for cell
    const Eigen::Map<const Eigen::Array<PetscInt, Eigen::Dynamic, 1>> dmap
        = dofmap.cell_dofs(cell.index());

    // TODO: Move gathering of coefficients outside of main assembly
    // loop
    // Update coefficients
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
      coefficients[i]->restrict(coeff_array.data() + offsets[i], cell,
                                coordinate_dofs);
    }

    // Tabulate element vector
    be.setZero(dmap.size());
    fn(be.data(), coeff_array.data(), coordinate_dofs.data(), local_facet, 1);

    // Add element vector to global vector
    for (Eigen::Index i = 0; i < dmap.size(); ++i)
      b[dmap[i]] += be[i];
  }
}
//-----------------------------------------------------------------------------
void fem::impl::assemble_interior_facets(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& L)
{
  throw std::runtime_error("Interior facet integrals not supported yet.");
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
      auto map1 = V1->dofmap()->index_map();
      assert(map1);
      const int crange
          = map1->block_size() * (map1->size_local() + map1->num_ghosts());
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
