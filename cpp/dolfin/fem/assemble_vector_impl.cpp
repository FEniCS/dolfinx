// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <petscis.h>

#include "DirichletBC.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "assemble_vector_impl.h"
#include <dolfin/common/types.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
void fem::impl::set_bc(Vec b,
                       std::vector<std::shared_ptr<const DirichletBC>> bcs,
                       double scale)
{
  PetscInt local_size;
  VecGetLocalSize(b, &local_size);
  PetscScalar* values = nullptr;
  VecGetArray(b, &values);
  assert(values);
  Eigen::Map<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> vec(values,
                                                                local_size);
  for (auto bc : bcs)
  {
    assert(bc);
    bc->set(vec, scale);
  }

  VecRestoreArray(b, &values);
}
//-----------------------------------------------------------------------------
void fem::impl::set_bc(Vec b,
                       std::vector<std::shared_ptr<const DirichletBC>> bcs,
                       const Vec x0, double scale)
{
  assert(b);
  assert(x0);
  PetscInt local_size_b, local_size_x0;
  VecGetLocalSize(b, &local_size_b);
  VecGetLocalSize(x0, &local_size_x0);

  if (local_size_b != local_size_x0)
    throw std::runtime_error("Size mismtach between b and x0 vectors.");
  PetscScalar* values_b;
  PetscScalar const* values_x0;
  VecGetArray(b, &values_b);
  VecGetArrayRead(x0, &values_x0);
  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> vec_b(values_b,
                                                                 local_size_b);
  const Eigen::Map<const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> vec_x0(
      values_x0, local_size_x0);
  for (auto bc : bcs)
  {
    assert(bc);
    bc->set(vec_b, vec_x0, scale);
  }

  VecRestoreArrayRead(x0, &values_x0);
  VecRestoreArray(b, &values_b);
}
//-----------------------------------------------------------------------------
void fem::impl::modify_bc(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& L,
    const std::vector<std::shared_ptr<const Form>> a,
    const std::vector<std::shared_ptr<const DirichletBC>> bcs,
    const Eigen::Ref<const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> x0,
    double scale)
{
  for (std::size_t i = 0; i < a.size(); ++i)
    fem::impl::modify_bc(b, *a[i], bcs, x0, scale);
}
//-----------------------------------------------------------------------------
void fem::impl::modify_bc(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& L,
    const std::vector<std::shared_ptr<const Form>> a,
    const std::vector<std::shared_ptr<const DirichletBC>> bcs, double scale)
{
  for (std::size_t i = 0; i < a.size(); ++i)
    fem::impl::modify_bc(b, *a[i], bcs, scale);
}
//-----------------------------------------------------------------------------
void fem::impl::assemble(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& L)
{
  // Get mesh from form
  assert(L.mesh());
  const mesh::Mesh& mesh = *L.mesh();

  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim);

  // Collect pointers to dof maps
  assert(L.function_space(0));
  assert(L.function_space(0)->dofmap());
  const fem::GenericDofMap& dofmap = *L.function_space(0)->dofmap();

  // Creat data structures used in assembly
  EigenRowArrayXXd coordinate_dofs;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;

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
    be.resize(dmap.size());
    be.setZero();

    // Compute local cell vector and add to global vector
    L.tabulate_tensor(be.data(), cell, coordinate_dofs);
    for (Eigen::Index i = 0; i < dmap.size(); ++i)
      b[dmap[i]] += be[i];
  }
}
//-----------------------------------------------------------------------------
void fem::impl::modify_bc(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b, const Form& a,
    std::vector<std::shared_ptr<const DirichletBC>> bcs, double scale)
{
  const Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> x0(0);
  fem::impl::_modify_bc(b, a, bcs, x0, scale);
}
//-----------------------------------------------------------------------------
void fem::impl::modify_bc(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b, const Form& a,
    std::vector<std::shared_ptr<const DirichletBC>> bcs,
    const Eigen::Ref<const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> x0,
    double scale)
{
  if (b.size() != x0.size())
  {
    throw std::runtime_error(
        "Vector size mismatch in modification for boundary conditions.");
  }

  fem::impl::_modify_bc(b, a, bcs, x0, scale);
}
//-----------------------------------------------------------------------------
void fem::impl::_modify_bc(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b, const Form& a,
    std::vector<std::shared_ptr<const DirichletBC>> bcs,
    const Eigen::Ref<const Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> x0,
    double scale)
{
  assert(a.rank() == 2);

  // Get mesh from form
  assert(a.mesh());
  const mesh::Mesh& mesh = *a.mesh();

  // Get bcs
  DirichletBC::Map boundary_values;
  for (std::size_t i = 0; i < bcs.size(); ++i)
  {
    assert(bcs[i]);
    assert(bcs[i]->function_space());
    if (a.function_space(1)->contains(*bcs[i]->function_space()))
      bcs[i]->get_boundary_values(boundary_values);
  }

  // Get dofmap for columns and rows of a
  assert(a.function_space(0));
  assert(a.function_space(0)->dofmap());
  assert(a.function_space(1));
  assert(a.function_space(1)->dofmap());
  const fem::GenericDofMap& dofmap0 = *a.function_space(0)->dofmap();
  const fem::GenericDofMap& dofmap1 = *a.function_space(1)->dofmap();

  // Data structures used in bc application
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;
  EigenRowArrayXXd coordinate_dofs;

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
    for (int i = 0; i < dmap1.size(); ++i)
    {
      const std::size_t ii = dmap1[i];
      if (boundary_values.find(ii) != boundary_values.end())
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
    Ae.resize(dmap0.size(), dmap1.size());
    Ae.setZero();
    a.tabulate_tensor(Ae.data(), cell, coordinate_dofs);

    // FIXME: Is this required?
    // Zero Dirichlet rows in Ae
    /*
    if (spaces[0] == spaces[1])
    {
      for (int i = 0; i < dmap0.size(); ++i)
      {
        const std::size_t ii = dmap0[i];
        auto bc = boundary_values.find(ii);
        if (bc != boundary_values.end())
          Ae.row(i).setZero();
      }
    }
    */

    // Size data structure for assembly
    be.resize(dmap0.size());
    be.setZero();

    for (int j = 0; j < dmap1.size(); ++j)
    {
      const std::size_t jj = dmap1[j];
      auto bc = boundary_values.find(jj);
      if (bc != boundary_values.end())
      {
        if (x0.rows() > 0)
          be -= Ae.col(j) * scale * (bc->second - x0[jj]);
        else
          be -= Ae.col(j) * scale * bc->second;
      }
    }

    for (Eigen::Index k = 0; k < dmap0.size(); ++k)
      b[dmap0[k]] += be[k];
  }
}
//-----------------------------------------------------------------------------
