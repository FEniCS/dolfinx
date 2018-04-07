// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "SpecialFunctions.h"
#include <dolfin/log/log.h>
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
void MeshCoordinates::eval(Eigen::Ref<EigenRowArrayXXd> values,
                           Eigen::Ref<const EigenRowArrayXXd> x,
                           const mesh::Cell& cell) const
{
  assert(_mesh);
  assert((unsigned int)x.cols() == _mesh->geometry().dim());

  values = x;
}
//-----------------------------------------------------------------------------
FacetArea::FacetArea(std::shared_ptr<const mesh::Mesh> mesh)
    : Expression({}), _mesh(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FacetArea::eval(Eigen::Ref<EigenRowArrayXXd> values,
                     Eigen::Ref<const EigenRowArrayXXd> x,
                     const mesh::Cell& cell) const
{
  assert(_mesh);

  for (unsigned int i = 0; i != x.rows(); ++i)
  {
    if (cell.local_facet >= 0)
      values(i, 0) = cell.facet_area(cell.local_facet);
    else
    {
      // not_on_boundary
      values(i, 0) = 0.0;
    }
  }
}
//-----------------------------------------------------------------------------
