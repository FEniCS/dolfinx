// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SpecialFunctions.h"
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>

using namespace dolfin::function;

//-----------------------------------------------------------------------------
MeshCoordinates::MeshCoordinates(std::shared_ptr<const mesh::Mesh> mesh)
    : Expression({mesh->geometry().dim()}), _mesh(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MeshCoordinates::eval(
    Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>>
        values,
    const Eigen::Ref<const EigenRowArrayXXd> x) const
{
  assert(_mesh);
  assert(x.cols() == _mesh->geometry().dim());
  values = x;
}
//-----------------------------------------------------------------------------
