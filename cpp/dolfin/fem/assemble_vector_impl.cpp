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
void fem::set_bc(Vec b, const Form& L,
                 std::vector<std::shared_ptr<const DirichletBC>> bcs,
                 double scale)
{
  PetscInt local_size;
  VecGetLocalSize(b, &local_size);
  PetscScalar* values;
  VecGetArray(b, &values);
  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> vec(values,
                                                               local_size);
  set_bc(vec, L, bcs, scale);
  VecRestoreArray(b, &values);
}
//-----------------------------------------------------------------------------
void fem::set_bc(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b,
                 const Form& L,
                 std::vector<std::shared_ptr<const DirichletBC>> bcs,
                 double scale)
{
  // FIXME: optimise this function

  auto V = L.function_space(0);
  Eigen::Array<PetscInt, Eigen::Dynamic, 1> indices;
  Eigen::Array<PetscScalar, Eigen::Dynamic, 1> values;
  for (std::size_t i = 0; i < bcs.size(); ++i)
  {
    assert(bcs[i]);
    assert(bcs[i]->function_space());
    if (V->contains(*bcs[i]->function_space()))
    {
      std::tie(indices, values) = bcs[i]->bcs();
      for (Eigen::Index j = 0; j < indices.size(); ++j)
      {
        // FIXME: this check is because DirichletBC::dofs include ghosts
        if (indices[j] < (PetscInt)b.size())
          b[indices[j]] = scale * values[j];
      }
    }
  }
}
//-----------------------------------------------------------------------------
void fem::assemble_ghosted(
    Vec b, const Form& L, const std::vector<std::shared_ptr<const Form>> a,
    const std::vector<std::shared_ptr<const DirichletBC>> bcs, double scale)
{
  Vec b_local;
  VecGhostGetLocalForm(b, &b_local);
  fem::assemble_local(b_local, L, a, bcs);

  // Restore ghosted form and update local (owned) entries that are
  // ghosts on other processes
  VecGhostRestoreLocalForm(b, &b_local);
  VecGhostUpdateBegin(b, ADD_VALUES, SCATTER_REVERSE);
  VecGhostUpdateEnd(b, ADD_VALUES, SCATTER_REVERSE);

  // Set boundary values (local only)
  set_bc(b, L, bcs, scale);
}
//-----------------------------------------------------------------------------
void fem::assemble_local(
    Vec& b, const Form& L, const std::vector<std::shared_ptr<const Form>> a,
    const std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  // FIXME: check that b is a local PETSc Vec

  // Wrap local PETSc Vec as an Eigen vector
  PetscInt size = 0;
  VecGetSize(b, &size);
  PetscScalar* b_array;
  VecGetArray(b, &b_array);
  Eigen::Map<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> bvec(b_array, size);

  //  Assemble and then modify for Dirichlet bcs  (b  <- b - A x_(bc))
  assemble_eigen(bvec, L, a, bcs);

  // Restore array
  VecRestoreArray(b, &b_array);
}
//-----------------------------------------------------------------------------
void fem::assemble_eigen(
    Eigen::Ref<Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1>> b, const Form& L,
    const std::vector<std::shared_ptr<const Form>> a,
    std::vector<std::shared_ptr<const DirichletBC>> bcs)
{
  // if (b.empty())
  //  init(b, L);

  // Get mesh from form
  assert(L.mesh());
  const mesh::Mesh& mesh = *L.mesh();

  const std::size_t tdim = mesh.topology().dim();
  mesh.init(tdim);

  // Collect pointers to dof maps
  auto dofmap = L.function_space(0)->dofmap();

  // Data structures used in assembly
  EigenRowArrayXXd coordinate_dofs;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;

  // Iterate over all cells
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Check that cell is not a ghost
    assert(!cell.is_ghost());

    // Get cell vertex coordinates
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get dof maps for cell
    auto dmap = dofmap->cell_dofs(cell.index());

    // Size data structure for assembly
    be.resize(dmap.size());
    be.setZero();

    // Compute cell matrix
    L.tabulate_tensor(be.data(), cell, coordinate_dofs);

    // Add to vector
    for (Eigen::Index i = 0; i < dmap.size(); ++i)
      b[dmap[i]] += be[i];
  }

  // Modify for any bcs
  for (std::size_t i = 0; i < a.size(); ++i)
    fem::modify_bc(b, *a[i], bcs);
}
//-----------------------------------------------------------------------------
void fem::modify_bc(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, 1>> b,
                    const Form& a,
                    std::vector<std::shared_ptr<const DirichletBC>> bcs)
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
    {
      bcs[i]->get_boundary_values(boundary_values);
      if (MPI::size(mesh.mpi_comm()) > 1
          and bcs[i]->method() != DirichletBC::Method::pointwise)
      {
        bcs[i]->gather(boundary_values);
      }
    }
  }

  // std::array<const function::FunctionSpace*, 2> spaces
  //    = {{a.function_space(0).get(), a.function_space(1).get()}};

  // Get dofmap for columns a a[i]
  auto dofmap0 = a.function_space(0)->dofmap();
  auto dofmap1 = a.function_space(1)->dofmap();

  Eigen::Matrix<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Ae;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> be;
  EigenRowArrayXXd coordinate_dofs;

  // Iterate over all cells
  for (auto& cell : mesh::MeshRange<mesh::Cell>(mesh))
  {
    // Check that cell is not a ghost
    assert(!cell.is_ghost());

    // Get dof maps for cell
    auto dmap1 = dofmap1->cell_dofs(cell.index());

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
    auto dmap0 = dofmap0->cell_dofs(cell.index());
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
        be -= Ae.col(j) * bc->second;
      }
    }

    for (Eigen::Index k = 0; k < dmap0.size(); ++k)
      b[dmap0[k]] += be[k];
  }
}
//-----------------------------------------------------------------------------
