// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Assembler.h"
#include "DirichletBC.h"
#include "Form.h"
#include "GenericDofMap.h"
#include "UFC.h"
#include "utils.h"
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshIterator.h>
#include <string>

using namespace dolfin;

//-----------------------------------------------------------------------------
fem::Assembler::Assembler(std::shared_ptr<const Form> a,
                          std::shared_ptr<const Form> L,
                          std::vector<std::shared_ptr<const DirichletBC>> bcs)
    : _a(a), _l(L), _bcs(bcs)
{
  // Check rank of forms
  if (a and a->rank() != 2)
  {
    throw std::runtime_error(
        "Expecting bilinear form (rank 2), but form has rank \'"
        + std::to_string(a->rank()) + "\'");
  }
  if (L and L->rank() != 1)
  {
    throw std::runtime_error(
        "Expecting linear form (rank 1), but form has rank \'"
        + std::to_string(L->rank()) + "\'");
  }
}
//-----------------------------------------------------------------------------
void fem::Assembler::assemble(PETScMatrix& A)
{
  assert(_a);
  if (A.empty())
    fem::init(A, *_a);

  // Get mesh from form
  assert(_a->mesh());
  const Mesh& mesh = *(_a->mesh());

  // FIXME: Remove UFC
  // Create data structures for local assembly data
  UFC ufc(*_a);

  const std::size_t D = mesh.topology().dim();
  mesh.init(D);

  // Collect pointers to dof maps
  std::array<const GenericDofMap*, 2> dofmaps
      = {{ufc.dolfin_form.function_space(0)->dofmap().get(),
          ufc.dolfin_form.function_space(1)->dofmap().get()}};

  ufc::cell ufc_cell;
  Eigen::MatrixXd coordinate_dofs;
  Eigen::MatrixXd Ae;

  // Get cell integral
  auto cell_integral = ufc.default_cell_integral;

  // Iterate over all cells
  for (auto& cell : MeshRange<Cell>(mesh))
  {
    // Check that cell is not a ghost
    assert(!cell.is_ghost());

    // Get cell vertex coordinates
    cell.get_coordinate_dofs(coordinate_dofs);

    // Get UFC cell data
    cell.get_cell_data(ufc_cell);

    // Update UFC data to current cell
    ufc.update(cell, coordinate_dofs, ufc_cell,
               cell_integral->enabled_coefficients());

    // Get dof maps for cell
    auto dmap0 = dofmaps[0]->cell_dofs(cell.index());
    auto dmap1 = dofmaps[1]->cell_dofs(cell.index());

    // Size data structure for assembly
    Ae.resize(dmap0.size(), dmap1.size());
    Ae.setZero();

    // Compute cell matrix
    cell_integral->tabulate_tensor(Ae.data(), ufc.w(), coordinate_dofs.data(),
                                   ufc_cell.orientation);

    A.add_local(Ae.data(), dmap0.size(), dmap0.data(), dmap1.size(),
                dmap1.data());
  }
}
//-----------------------------------------------------------------------------
void fem::Assembler::assemble(PETScVector& b)
{
  assert(_l);
  if (b.empty())
    fem::init(b, *_l);

  // Get mesh from form
  assert(_l->mesh());
  const Mesh& mesh = *(_l->mesh());

  // FIXME: Remove UFC
  // Create data structures for local assembly data
  UFC ufc(*_l);

  throw std::runtime_error("Not implemented");
}
//-----------------------------------------------------------------------------
